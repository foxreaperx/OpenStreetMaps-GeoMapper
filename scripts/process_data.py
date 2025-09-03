import geopandas as gpd
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString, box
from shapely.ops import nearest_points, linemerge, unary_union, snap
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree
import logging
import time
from shapely.errors import GEOSException
from contextlib import contextmanager

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@contextmanager
# Purpose: Context manager to log the start and end of a processing step with elapsed time.
# Inputs:
# - label (str): Human readable step label to include in log messages.
# Outputs:
# - None. Produces INFO log lines when entering and leaving the context.
def log_step(label: str):
    """Log start/end and wall time of a processing step."""
    logger.info(f"[START] {label}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info(f"[END]   {label} in {dt:.2f}s")

# --- Load config ---
with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

LOCAL_CRS = CONFIG.get("local_crs", "EPSG:32610")  # UTM zone for SF
WGS84 = "EPSG:4326"

# Defaults
DEFAULTS = {
    "buffers": {
        "road_width": 10.0,
        "sidewalk_width": 2.0,
        "crosswalk_width": 3.0
    },
    "per_lane_width": 3.2,
    "parking_lane_width": 2.3,
    "hierarchy_widths": {
        "motorway": 28.0, "trunk": 22.0, "primary": 18.0, "secondary": 14.0,
        "tertiary": 12.0, "residential": 10.0, "unclassified": 9.0, "service": 8.0,
        "living_street": 8.0, "footway": 4.0, "path": 4.0, "pedestrian": 10.0,
        "track": 6.0, "cycleway": 4.0, "steps": 3.0, "construction": 10.0
    }
}

BUFFERS = {**DEFAULTS["buffers"], **CONFIG.get("buffers", {})}
LANE_W  = float(CONFIG.get("per_lane_width", DEFAULTS["per_lane_width"]))
PARK_W  = float(CONFIG.get("parking_lane_width", DEFAULTS["parking_lane_width"]))
HIER    = {**DEFAULTS["hierarchy_widths"], **CONFIG.get("hierarchy_widths", {})}

MIN_HALF_WIDTH = 1.2  # meters

# --- Crosswalk placement rules (driveway/aisle guards) ---
CROSSWALK_RULES = {
    "snap_tolerance_m": 12.0,      # max distance to snap point to a road
    "intersection_radius_m": 8.0,  # radius to count nearby road segments (degree)
    "min_deg_for_point": 2,        # require at least this many segments nearby
    "min_nearest_road_len_m": 20.0,# drop points on very short stubs (driveways)
    "allow_service": False,        # if False, exclude highway=service entirely
    "service_exclusions": {"driveway", "parking_aisle", "alley"}  # used by alt filters
}

DRIVABLE_HIGHWAYS = {
    "motorway","trunk","primary","secondary","tertiary",
    "residential","unclassified","living_street","service"  # can be excluded by allow_service
}

# --- Crosswalk “connect to both sidewalks” tuning ---
CROSSWALK_CONNECT = {
    "curb_overlap_m": 0.75,         # starting curb overlap into sidewalks
    "max_curb_overlap_m": 1.8,      # try up to this much overlap if needed
    "curb_overlap_step_m": 0.25,    # step size while searching
    "grow_steps": (1.2, 1.6, 2.0, 2.6, 3.2, 3.8, 4.4, 5.0, 5.6),  # longer options
    "fallback_boost_pct": 0.12,     # pre-boost curb-to-curb span by +12% (helps skewed polygons)
}

CROSSWALK_SHAPE_LIMITS = {
    "max_length_multiplier": 1.8,      # cap L <= 1.8 × curb-to-curb span
    "max_abs_length_m": 40.0,          # and never longer than 40 m
    "max_axis_misalignment_deg": 25.0, # must be within 25° of road normal
    "max_aspect_ratio": 14.0           # reject super long slivers (L/S)
}

CROSSWALK_FALLBACKS = {
    "enable_if_no_sidewalks": True,   # accept crosswalks without sidewalks present
    "accept_if_two_curb_hits": True,  # require crosswalk to reach both road edges
    "min_road_overlap_m2": 1.2,       # minimum area overlapped with road for acceptance
    "max_center_to_road_m": 16.0,     # seed→road proximity guard
    "connector_search_m": 12.0,       # how far to look for a sidewalk/ped boundary
    "connector_width_m": 1.5,         # connector sidewalk width
}

# -------------------- Utilities --------------------
# Purpose: Determine whether a rectangle crosses a road polygon from curb to curb.
# Inputs:
# - rect (Polygon): Candidate rectangle.
# - roads_union_poly (Polygon/MultiPolygon): Unary union of street polygons in LOCAL_CRS.
# Outputs:
# - bool: True when the rectangle intersects the road boundary in two or more distinct places.
def _touches_road_both_curbs(rect, roads_union_poly) -> bool:
    if not rect or rect.is_empty or not roads_union_poly or roads_union_poly.is_empty:
        return False
    boundary = roads_union_poly.boundary
    inter = rect.intersection(boundary)
    if inter.is_empty:
        return False
    geoms = []
    if inter.geom_type == "LineString":
        geoms = [inter]
    elif inter.geom_type == "MultiLineString":
        geoms = list(inter.geoms)
    else:
        geoms = [g for g in getattr(inter, "geoms", []) if g.geom_type in ("LineString","MultiLineString")]
    return len(geoms) >= 2  # hits two distinct curb lines

# Purpose: Return the midpoints of the two shortest edges of a rectangle.
# Inputs:
# - rect (Polygon): Minimum rotated rectangle or similar four-sided polygon in LOCAL_CRS.
# Outputs:
# - list[Point]: Up to two midpoints ordered from the shorter edge to the next.
def _short_edge_midpoints(rect):
    if rect is None or rect.is_empty:
        return []
    try:
        coords = list(rect.exterior.coords)
    except Exception:
        return []
    edges = []
    for i in range(4):
        p = Point(coords[i]); q = Point(coords[i+1])
        L = p.distance(q)
        edges.append((L, p, q))
    edges.sort(key=lambda x: x[0])  # shortest first
    mids = []
    for k in range(min(2, len(edges))):
        p, q = edges[k][1], edges[k][2]
        mids.append(Point((p.x+q.x)/2.0, (p.y+q.y)/2.0))
    return mids

# Purpose: Compute the long and short side lengths of a polygon’s minimum rotated rectangle.
# Inputs:
# - poly (Polygon or MultiPolygon): Input polygon in any CRS.
# Outputs:
# - tuple[float, float]: (long_side, short_side) in CRS units.
def _mrr_lengths(poly):
    """Return (L, S) = long and short side lengths of min rotated rect."""
    if poly is None or poly.is_empty:
        return (0.0, 0.0)
    if poly.geom_type == "MultiPolygon":
        poly = max(list(poly.geoms), key=lambda g: g.area)
    mrr = poly.minimum_rotated_rectangle
    cs = list(mrr.exterior.coords)
    sides = [np.hypot(cs[i+1][0]-cs[i][0], cs[i+1][1]-cs[i][1]) for i in range(4)]
    L = max(sides); S = min(sides)
    return (float(L), float(S))

# Purpose: Validate a crosswalk rectangle against orientation, length, and aspect-ratio limits.
# Inputs:
# - rect (Polygon): Candidate rectangle.
# - pt (Point): Seed point on the associated road line.
# - road_line (LineString): Road centerline in LOCAL_CRS.
# - span (float): Curb-to-curb distance in meters.
# - stripe_w (float): Crosswalk width in meters.
# Outputs:
# - bool: True when the rectangle meets all geometric constraints.
def _is_good_crosswalk_rect(rect, pt, road_line, span, stripe_w):
    """Gate stripes by orientation, length, and aspect ratio."""
    if rect is None or rect.is_empty or road_line is None or road_line.is_empty:
        return False
    road_bear = _bearing_at_point_on_line(pt, road_line)
    major_deg = _poly_major_axis_deg(rect)
    if road_bear is None or major_deg is None:
        return False
    normal_deg = (road_bear + 90.0) % 180.0
    mis = _acute_diff_deg(major_deg, normal_deg)
    if mis > float(CROSSWALK_SHAPE_LIMITS["max_axis_misalignment_deg"]):
        return False
    L, S = _mrr_lengths(rect)
    if S <= 0: return False
    if L / max(S, 1e-6) > float(CROSSWALK_SHAPE_LIMITS["max_aspect_ratio"]):
        return False
    cap = min(float(CROSSWALK_SHAPE_LIMITS["max_length_multiplier"]) * float(span),
              float(CROSSWALK_SHAPE_LIMITS["max_abs_length_m"]))
    return L <= cap

# Purpose: Check whether an OSM 'highway' value is considered drivable for crosswalk placement.
# Inputs:
# - hw (str): Highway tag value (e.g., 'primary','residential','service').
# Outputs:
# - bool: True when the tag is in the configured drivable set and not excluded by service rules.
def _is_drivable_highway(hw: str) -> bool:
    hw = (hw or "").lower()
    if hw not in DRIVABLE_HIGHWAYS:
        return False
    if hw == "service" and not CROSSWALK_RULES["allow_service"]:
        return False
    return True

# Purpose: Approximate the road network degree at a point by counting nearby road segments.
# Inputs:
# - pt (Point): Location in LOCAL_CRS.
# - roads_local (GeoDataFrame): Road centerlines in LOCAL_CRS.
# - radius (float): Search radius in meters.
# Outputs:
# - int: Count of segments whose distance to pt is less than or equal to radius.
def _local_road_degree(pt: Point, roads_local: gpd.GeoDataFrame, radius: float) -> int:
    """Approximate 'intersection-ness': count road segments whose geometry is within radius of the point."""
    if roads_local.empty:
        return 0
    buf = pt.buffer(radius)
    idxs = list(roads_local.sindex.intersection(buf.bounds))
    if not idxs:
        return 0
    cand = roads_local.iloc[idxs]
    return int((cand.distance(pt) <= radius).sum())

# Purpose: Drop polygons below a minimum area threshold after projecting to a local metric CRS.
# Inputs:
# - gdf (GeoDataFrame): Polygonal layer in any CRS.
# - min_area_m2 (float): Minimum area in square meters to keep.
# - crs_local (str): Local projected CRS used for measuring area.
# Outputs:
# - GeoDataFrame: Filtered polygons returned in the original CRS.
def remove_small_polygons(gdf: gpd.GeoDataFrame, min_area_m2=0.5, crs_local: str = None):
    if gdf.empty:
        return gdf
    crs_local = crs_local or LOCAL_CRS
    local = gdf.to_crs(crs_local).copy()
    keep = local.geometry.area >= float(min_area_m2)
    local = local.loc[keep].copy()
    return local.to_crs(WGS84)

# Purpose: Morphologically close narrow gaps within sidewalk polygons by buffer in and buffer out.
# Inputs:
# - gdf (GeoDataFrame): Sidewalk polygons in any CRS.
# - gap_m (float): Gap size in meters to close (buffer distance applied in LOCAL_CRS).
# - crs_local (str): Local projected CRS in meters.
# Outputs:
# - GeoDataFrame: Cleaned sidewalk polygons returned in the original CRS.
def close_sidewalk_gaps(sidewalks_poly_ll: gpd.GeoDataFrame, gap_m=0.18) -> gpd.GeoDataFrame:
    if sidewalks_poly_ll.empty:
        return sidewalks_poly_ll
    sw_l = sidewalks_poly_ll.to_crs(LOCAL_CRS).copy()
    sw_l["geometry"] = sw_l.buffer(gap_m).buffer(-gap_m)
    sw_l["geometry"] = sw_l.geometry.apply(_fix_geom)
    return sw_l.to_crs(WGS84)

# Purpose: Keep polygonal features only and dissolve multipart polygons to simple polygons where applicable.
# Inputs:
# - gdf (GeoDataFrame): Mixed-geometry layer.
# Outputs:
# - GeoDataFrame: Only Polygon/MultiPolygon geometries, invalids fixed with buffer(0), in the original CRS.
def ensure_polygons_only(gdf: gpd.GeoDataFrame, *, log_label: str = "layer") -> gpd.GeoDataFrame:
    """Return a copy containing only Polygon geometries (explode MultiPolygons, drop others)."""
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=(gdf.crs if gdf is not None else WGS84))
    fixed = gdf.copy()
    fixed["geometry"] = fixed.geometry.apply(_fix_geom)
    fixed = fixed[~fixed.geometry.is_empty]
    try:
        exploded = fixed.explode(index_parts=True, ignore_index=True)
    except TypeError:
        exploded = fixed.explode().reset_index(drop=True)
    poly_only = exploded[exploded.geometry.geom_type == "Polygon"].copy()
    poly_only = poly_only[~poly_only.geometry.is_empty]
    dropped = len(exploded) - len(poly_only)
    logger.info(f"{log_label}: kept {len(poly_only)} polygons (dropped {dropped} non-polygons).")
    return poly_only

# Purpose: Return up to 'maxn' road segments within a radius of a point, ordered by distance.
# Inputs:
# - pt (Point): Query point in LOCAL_CRS.
# - roads_local (GeoDataFrame): Road centerlines in LOCAL_CRS with a spatial index.
# - radius_m (float): Search radius in meters.
# - maxn (int): Maximum number of candidates to return.
# Outputs:
# - list[pandas.NamedTuple]: Road rows with attributes, closest first.
def _nearby_roads(pt: Point, roads_local: gpd.GeoDataFrame, radius_m=20.0, maxn=6):
    if roads_local.empty:
        return []
    cand_idx = list(roads_local.sindex.intersection(pt.buffer(radius_m).bounds))
    if not cand_idx:
        return []
    cand = roads_local.iloc[cand_idx].copy()
    cand["__dist"] = cand.geometry.distance(pt)
    cand = cand.sort_values("__dist").head(maxn)
    return list(cand.itertuples(index=True))

# Purpose: Score a candidate crosswalk rectangle using curb or sidewalk connectivity, closeness to target span, and road overlap.
# Inputs:
# - rect (Polygon): Candidate rectangle in LOCAL_CRS.
# - roads_union_poly (Polygon/MultiPolygon): Road-union polygon in LOCAL_CRS.
# - sw_union (geometry): Union of sidewalk and pedestrian-zone boundaries in LOCAL_CRS.
# - span (float): Target curb-to-curb width in meters.
# - attr_w (float): Attribute-based width estimate in meters used for closeness.
# Outputs:
# - tuple[int,float,float]: (touches_score, closeness_score, road_overlap_area).
def _score_stripe(rect, roads_union_poly, sw_union, span=None, attr_w=None):
    if rect is None or rect.is_empty:
        return (-1, -1e9, 0.0)
    road_overlap = 0.0
    if roads_union_poly is not None and not roads_union_poly.is_empty:
        road_overlap = rect.intersection(roads_union_poly).area
    touches_sw = _touches_sidewalk_both_sides(rect, sw_union)
    touches_curbs = _touches_road_both_curbs(rect, roads_union_poly)
    touches_score = 2 if touches_sw else (1 if touches_curbs else 0)
    closeness = -1e9
    if span is not None and attr_w is not None and attr_w > 0:
        closeness = -abs(np.log((float(span) + 1e-6) / (float(attr_w) + 1e-6)))
    return (touches_score, float(closeness), float(road_overlap))

# Purpose: Parse an OSM 'lanes' value to a numeric lane count.
# Inputs:
# - x (any): Raw lanes attribute. Accepts single numeric strings or semicolon-separated lists.
# Outputs:
# - float or numpy.nan: Parsed lane count (mean for lists) or NaN when parsing fails.
def _parse_lanes(x):
    try:
        if x is None: return np.nan
        s = str(x).strip()
        if ";" in s:
            vals = [float(v) for v in s.split(";") if v.strip() != ""]
            return float(np.nanmean(vals)) if vals else np.nan
        return float(s)
    except Exception:
        return np.nan

# Purpose: Estimate total curb-to-curb width from highway hierarchy and lane count attributes.
# Inputs:
# - hw (str): OSM 'highway' value.
# - lanes (any): OSM 'lanes' value.
# Outputs:
# - float: Width in meters computed from configured hierarchy widths and per-lane plus parking assumptions.
def _infer_total_width_from_attrs(hw: str, lanes) -> float:
    base = float(HIER.get((hw or "").lower(), BUFFERS["road_width"]))
    lanes_val = _parse_lanes(lanes)
    return max(base, lanes_val * LANE_W + 2 * PARK_W) if not np.isnan(lanes_val) else base


# Purpose: Row-wise helper to compute total road width with a minimum when no nearest-attribute join occurred.
# Inputs:
# - row (pandas.Series): Contains 'highway', 'lanes', and optionally 'nearest_dist' from the join.
# Outputs:
# - float: Width in meters.
def infer_total_road_width(row) -> float:
    base = _infer_total_width_from_attrs(row.get("highway"), row.get("lanes"))
    # If we *don't* have a nearest attribute join, keep a sane lower bound
    return base if not np.isnan(row.get("nearest_dist", np.nan)) else max(base, 12.0)

# Purpose: Sample distances from a line to a polygon and return a robust percentile statistic.
# Inputs:
# - line (LineString): Query line in LOCAL_CRS.
# - poly (Polygon or MultiPolygon): Target polygon in the same CRS.
# - n (int): Number of evenly spaced samples along the line.
# - q (float): Percentile in [0,100].
# Outputs:
# - float: The q-th percentile of point-to-polygon distances in CRS units.
def _sample_line_dist_percentile(line: LineString, target_geom, n=8, q=0.25) -> float:
    if line.length == 0 or target_geom.is_empty:
        return np.inf
    dists = []
    for i in range(n):
        pt = line.interpolate((i + 0.5) / n, normalized=True)
        dists.append(pt.distance(target_geom))
    return float(np.percentile(dists, q * 100.0))

# Purpose: Compute a road half-buffer width that preserves a minimum clear distance to buildings.
# Inputs:
# - geom (shapely.geometry.base.BaseGeometry): The road centerline or polygon in LOCAL_CRS units.
# - building_union (shapely.geometry.base.BaseGeometry): Unary union of building footprints in the same CRS.
# - desired_half (float): Desired half width in meters.
# - min_sidewalk_clear (float): Minimum required clearance to buildings in meters.
# Outputs:
# - float: Half width in meters to use for buffering the road so as to respect the building setback.
def adaptive_halfwidth(geom, building_union, desired_half: float, min_sidewalk_clear: float) -> float:
    d = _sample_line_dist_percentile(geom, building_union, n=8, q=0.25)
    if np.isinf(d):
        return max(desired_half, MIN_HALF_WIDTH)
    allowed = max(MIN_HALF_WIDTH, d - min_sidewalk_clear)
    return max(MIN_HALF_WIDTH, min(desired_half, allowed))

# Purpose: Return a copy of the GeoDataFrame filtered to a single geometry family.
# Inputs:
# - gdf (GeoDataFrame): Any geometry type.
# - geom_type (str): Geometry type to keep. One of 'Point','LineString','Polygon' (case-sensitive).
# Outputs:
# - GeoDataFrame: Same CRS as input, containing only features of the requested geometry family.
def ensure_geom_type(gdf: gpd.GeoDataFrame, default_type: str, crs=WGS84) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with ['geometry','type'] present. If missing/empty, create safely."""
    if gdf is None or not hasattr(gdf, "columns") or len(gdf.columns) == 0:
        return gpd.GeoDataFrame({"type": []}, geometry=[], crs=crs)[["geometry", "type"]]
    out = gdf.copy()
    if "type" not in out.columns:
        out = out.assign(type=default_type)
    if out.crs is None:
        out.set_crs(crs, inplace=True)
    return out[["geometry", "type"]]

# --- Validity helpers & snap-round clean ---
try:
    from shapely.validation import make_valid as _make_valid
    def _fix_geom(g):
        if g is None or g.is_empty: return g
        return _make_valid(g) if not g.is_valid else g
except Exception:
    def _fix_geom(g):
        if g is None or g.is_empty: return g
        return g.buffer(0)

# Purpose: Clean an entire GeoDataFrame’s geometries to remove micro slivers and invalid shapes.
# Inputs:
# - gdf (GeoDataFrame): Input layer.
# - grid_size (float): Snap-round grid size to apply during make-valid, in CRS units.
# - crs_local (str): Local projected CRS to use while cleaning when input is in WGS84.
# Outputs:
# - GeoDataFrame: Cleaned geometries, returned in the original CRS.
def clean_geoms(gdf: gpd.GeoDataFrame, snap_round_m: float = 0.02, crs_local: str = None) -> gpd.GeoDataFrame:
    """Validate and snap-round polygons/lines to remove slivers/micro-gaps."""
    if gdf.empty: return gdf
    crs_local = crs_local or gdf.crs
    with log_step("Snap-round clean"):
        local = gdf.to_crs(crs_local).copy()
        local["geometry"] = local.geometry.apply(_fix_geom)
        local["geometry"] = local.buffer(snap_round_m).buffer(-snap_round_m)
        local["geometry"] = local.geometry.apply(_fix_geom)
        return local.to_crs(WGS84)

# Purpose: Produce an empty geometry of the same family as the input geometry.
# Inputs:
# - geom (shapely geometry): Source geometry that determines the family (line-like or area-like).
# Outputs:
# - shapely geometry: Empty LineString for line-like inputs, empty Polygon for area-like inputs, empty GeometryCollection otherwise.
def _empty_like(g):
    try:
        if isinstance(g, LineString):
            return LineString()
        if isinstance(g, (Polygon, MultiPolygon)):
            return Polygon()
    except Exception:
        pass
    return LineString()

# Purpose: Robust intersection that retries with increasing snap-rounding and safe fallbacks.
# Inputs:
# - a (shapely geometry): First operand.
# - b (shapely geometry): Second operand.
# - label (str): Label used in logs.
# - grid_sizes (tuple[float|None,...]): Sequence of grid sizes to try during intersection.
# - fallback (str): 'a' to return operand A if all attempts fail, 'empty' to return an empty geometry.
# Outputs:
# - shapely geometry: The intersection result, or the selected fallback.
def _safe_intersection(a, b, *, label="", grid_sizes=(0.02, 0.05, 0.1, None), fallback="empty"):
    """Robust intersection with make-valid + snap-round fallbacks."""
    if a is None or b is None:
        return _empty_like(a)
    if getattr(a, "is_empty", False) or getattr(b, "is_empty", False):
        return _empty_like(a)
    A = _fix_geom(a)
    B = _fix_geom(b)
    for gs in grid_sizes:
        try:
            return (A.intersection(B, grid_size=gs) if gs is not None else A.intersection(B))
        except GEOSException:
            pass
        except Exception:
            pass
    logger.warning(f"_safe_intersection fallback for {label}")
    return A if fallback == "a" else _empty_like(a)

# Purpose: Measure the length of an intersection result, choosing the segment nearest to a reference point.
# Inputs:
# - geom (shapely geometry): Intersection result that be a LineString, MultiLineString, or collection.
# - pt (Point): Reference point used to select the nearest piece.
# Outputs:
# - float: Length in CRS units of the selected line segment, or 0.0 if none.
def _length_from_intersection(geom, pt):
    """Return length of line intersection; pick the piece closest to pt."""
    if geom.is_empty:
        return 0.0
    if isinstance(geom, LineString):
        return float(geom.length)
    if isinstance(geom, MultiLineString):
        parts = list(geom.geoms)
        if not parts:
            return 0.0
        parts.sort(key=lambda g: g.distance(pt))
        return float(parts[0].length)
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, LineString)]
    if not parts:
        return 0.0
    parts.sort(key=lambda g: g.distance(pt))
    return float(parts[0].length)

# Purpose: Buffer with tolerance to minor geometry invalidities.
# Inputs:
# - geom (shapely geometry): Geometry to buffer.
# - dist (float): Buffer distance in CRS units.
# - **kwargs: Forwarded buffer options (cap_style, join_style, etc.).
# Outputs:
# - shapely geometry or None: Buffered geometry, or None if buffering fails after cleaning.
def _safe_buffer(geom, dist, **kwargs):
    """Buffer that tolerates slightly invalid geometry."""
    if geom is None or getattr(geom, "is_empty", True):
        return None
    try:
        return geom.buffer(dist, **kwargs)
    except Exception:
        try:
            return _fix_geom(geom).buffer(dist, **kwargs)
        except Exception:
            logger.debug("safe_buffer failed; returning None")
            return None

# Purpose: Precompute and cache buffered versions of the road-union polygon for a range of overlap distances.
# Inputs:
# - roads_union_poly (Polygon or MultiPolygon): Unary-union of road polygons in LOCAL_CRS.
# - start_overlap (float): Smallest overlap distance in meters.
# - max_overlap (float): Largest overlap distance in meters.
# - step_overlap (float): Step between consecutive overlap distances in meters.
# Outputs:
# - dict[float, shapely geometry]: Mapping from overlap distance to buffered polygon to reuse in loops.
def _build_road_buffer_cache(roads_union_poly, start_overlap, max_overlap, step_overlap):
    """Pre-build road union buffers at overlap distances to avoid repeated buffering in loops."""
    if roads_union_poly is None or getattr(roads_union_poly, "is_empty", True):
        logger.info("Road union polygon is empty; buffer cache skipped.")
        return {}
    overlaps = []
    ov = float(start_overlap)
    while ov <= float(max_overlap) + 1e-9:
        overlaps.append(round(ov, 3))
        ov += float(step_overlap)
    logger.info(f"Building road buffer cache for {len(overlaps)} overlap values: {overlaps}")
    cache = {}
    t0 = time.perf_counter()
    for key in overlaps:
        cache[key] = _safe_buffer(roads_union_poly, key, cap_style=2, join_style=2)
    logger.info(f"Built road buffer cache in {time.perf_counter()-t0:.2f}s")
    return cache

# Purpose: Log the approximate width and height of a latitude/longitude bounding box.
# Inputs:
# - north (float), south (float), east (float), west (float): Bounding box edges in degrees.
# Outputs:
# - None. Writes an INFO log line with the estimated size in kilometers.
def describe_bbox(north, south, east, west):
    lat_mid = (north + south)/2
    h_km = (north - south) * 111.32
    w_km = (east - west) * 111.32 * np.cos(np.deg2rad(lat_mid))
    logger.info(f"BBox ~ {w_km:.2f} km × {h_km:.2f} km, center=({(north+south)/2:.5f}, {(east+west)/2:.5f})")
    if h_km < 1 or w_km < 1:
        logger.warning("BBox is very small (<1 km); risk of empty clips.")

# Purpose: Clip a layer by an axis-aligned bounding box in a target CRS.
# Inputs:
# - gdf (GeoDataFrame): Layer to clip.
# - west (float), south (float), east (float), north (float): Bounding box coordinates in target CRS units.
# - crs (str): Target CRS of the bbox extents. Default is 'EPSG:4326' (WGS84).
# Outputs:
# - GeoDataFrame: Subset of input features intersecting the bbox, in the given CRS.
def clip_bbox(gdf, west, south, east, north, crs="EPSG:4326"):
    assert east > west and north > south, "BBox must satisfy east>west and north>south"
    describe_bbox(north, south, east, west)
    bbox = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs=crs)
    return gpd.clip(gdf.to_crs(crs), bbox)

# -------------------- I/O helper --------------------
# Purpose: Read a geospatial file with CRS normalization and robust logging.
# Inputs:
# - path (Path): File path to load (GeoPackage, GeoJSON, Shapefile, etc.).
# Outputs:
# - GeoDataFrame: Loaded layer with CRS set to WGS84 if the file had no CRS.
def safe_read(path: Path):
    if not path.exists():
        logger.warning(f"File {path} does not exist. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    with log_step(f"Read {path.name}"):
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(WGS84, inplace=True)
    logger.info(f"Loaded {path} with {len(gdf)} features.")
    return gdf

# -------------------- Street Fixer --------------------
# Purpose: Snap, merge, and re-segment raw OSM road centerlines to produce clean linework with attributes.
# Processing steps in LOCAL_CRS units:
# - unary_union to dissolve overlapping lines
# - snap to self to close tiny gaps within a tolerance
# - linemerge to build topologically coherent lines
# - explode to individual segments and drop empties
# - attach nearest 'highway' and 'lanes' from original lines to each fixed segment
# Inputs:
# - roads (GeoDataFrame): Raw road centerlines in WGS84.
# - tolerance (float): Snap tolerance in meters in LOCAL_CRS.
# Outputs:
# - GeoDataFrame: Cleaned road segments in WGS84 with columns ['geometry','highway','lanes','nearest_dist'].
def fix_street_geometries(roads: gpd.GeoDataFrame, tolerance=2.0) -> gpd.GeoDataFrame:
    if roads.empty:
        return roads
    with log_step("Fix road linework (unary_union → snap → linemerge)"):
        roads_attr = roads.copy().to_crs(LOCAL_CRS)
        local = roads.to_crs(LOCAL_CRS)
        tol = max(0.5, float(tolerance))
        t0 = time.perf_counter()
        merged = unary_union(local.geometry)
        logger.info(f"unary_union done in {time.perf_counter()-t0:.2f}s")
        t1 = time.perf_counter()
        snapped = snap(merged, merged, tol)
        logger.info(f"snap done in {time.perf_counter()-t1:.2f}s (tolerance={tol})")
        t2 = time.perf_counter()
        fixed = linemerge(snapped)
        logger.info(f"linemerge done in {time.perf_counter()-t2:.2f}s")
    if isinstance(fixed, LineString):
        geoms = [fixed]
    elif isinstance(fixed, MultiLineString):
        geoms = list(fixed.geoms)
    else:
        geoms = []
    fixed_gdf = gpd.GeoDataFrame(geometry=geoms, crs=LOCAL_CRS)
    if fixed_gdf.empty:
        logger.warning("fix_street_geometries produced 0 segments.")
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    try:
        with log_step("Restore nearest attributes to fixed streets"):
            fixed_centroids = fixed_gdf.copy()
            fixed_centroids["geometry"] = fixed_centroids.centroid
            candidate_attrs = ["highway", "lanes"]
            attrs = [c for c in candidate_attrs if c in roads_attr.columns]
            fixed_gdf["nearest_dist"] = np.nan
            if attrs:
                join_source = roads_attr[attrs + ["geometry"]].copy()
                sjoined = gpd.sjoin_nearest(
                    fixed_centroids, join_source, how="left", distance_col="nearest_dist"
                ).drop(columns=["index_right"])
                for col in attrs + ["nearest_dist"]:
                    fixed_gdf[col] = sjoined[col].values
            else:
                logger.info("No highway/lanes columns present in raw; using width fallbacks.")
    except Exception as e:
        logger.warning(f"Nearest attr join failed (optional): {e}")
    out = fixed_gdf.to_crs(WGS84)
    logger.info(f"Fixed streets: input {len(roads)} → output {len(out)} segments")
    return out

# -------------------- Road Buffering --------------------
# Purpose: Buffer road centerlines to polygons using highway-specific base widths with context-sensitive adjustments.
# Workflow in LOCAL_CRS units:
# - Estimate total widths per segment from attributes and nearby geometry.
# - Enforce a minimum centerline length along the sampling corridor.
# - Apply adaptive_halfwidth near buildings to preserve clearances.
# - Buffer each segment and dissolve to obtain road-asphalt polygons.
# Inputs:
# - roads_ll (GeoDataFrame): Road centerlines in WGS84 with 'highway' and 'lanes' where available.
# - buildings_ll (GeoDataFrame): Building footprints in WGS84.
# - min_centerline_len (float): Minimum usable centerline length in meters for buffering decisions.
# Outputs:
# - GeoDataFrame: Road polygons in WGS84 with column 'type' set to 'street'.
def buffer_roads_hierarchical(roads_ll: gpd.GeoDataFrame, buildings_ll: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if roads_ll.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    with log_step("Road buffering"):
        roads = roads_ll.to_crs(LOCAL_CRS).copy()
        buildings = buildings_ll.to_crs(LOCAL_CRS).copy()
        t0 = time.perf_counter()
        b_union = buildings.geometry.union_all() if not buildings.empty else Polygon()
        logger.info(f"Building union prepared in {time.perf_counter()-t0:.2f}s (buildings={len(buildings)})")
        desired_total = roads.apply(infer_total_road_width, axis=1).astype(float)
        desired_half = desired_total / 2.0
        min_sidewalk_clear = float(BUFFERS["sidewalk_width"])
        buffered_geoms = []
        dropped_idx = []
        total = len(roads)
        logger.info(
            f"Buffering {total} road segments "
            f"(median target half-width={np.nanmedian(desired_half):.2f} m, "
            f"min sidewalk clear={min_sidewalk_clear:.2f} m)"
        )
        start = time.perf_counter()
        last_report = start
        report_every = max(50, total // 20)  # ~5% steps, min 50
        for i, ((idx, line), half) in enumerate(zip(roads.geometry.items(), desired_half)):
            if line is None or line.is_empty:
                buffered_geoms.append(None); dropped_idx.append(idx); continue
            try:
                half_eff = adaptive_halfwidth(line, b_union, half, min_sidewalk_clear)
                if half_eff <= 0:
                    half_eff = MIN_HALF_WIDTH
                buff = line.buffer(half_eff, cap_style=2, join_style=2)
                if buff.is_empty or (hasattr(buff, "area") and buff.area == 0):
                    buffered_geoms.append(None); dropped_idx.append(idx)
                else:
                    buffered_geoms.append(buff)
            except Exception:
                buffered_geoms.append(None); dropped_idx.append(idx)
            if ((i + 1) % report_every == 0) or (time.perf_counter() - last_report > 5.0):
                pct = 100.0 * (i + 1) / total
                elapsed = time.perf_counter() - start
                logger.info(f"  … buffered {i+1}/{total} ({pct:.1f}%) in {elapsed:.1f}s")
                last_report = time.perf_counter()
        out = gpd.GeoDataFrame(geometry=buffered_geoms, crs=LOCAL_CRS)
        out = out.dropna(subset=["geometry"]).copy()
        out = out[out.geometry.is_valid & ~out.geometry.is_empty].copy()
        out["type"] = "street"
        if dropped_idx:
            logger.warning(
                f"Dropped {len(dropped_idx)} road segments due to zero/invalid buffers "
                f"(debug_dropped_roads.geojson)."
            )
            try:
                roads.loc[dropped_idx].to_crs(WGS84).to_file("debug_dropped_roads.geojson", driver="GeoJSON")
            except Exception:
                pass
    return out.to_crs(WGS84)

# -------------------- Orientation helpers --------------------
# Purpose: Compute the acute angular difference between two orientations modulo 180 degrees.
# Inputs:
# - a (float): Angle in degrees.
# - b (float): Angle in degrees.
# Outputs:
# - float: Smallest difference in degrees within [0, 90].
def _acute_diff_deg(a, b):
    """Smallest angular difference (0..90) for directions modulo 180°."""
    d = abs((a - b) % 180.0)
    return d if d <= 90.0 else 180.0 - d

# Purpose: Compute the local tangent bearing (0 to 180 degrees) of a road line at the closest location to a point.
# Inputs:
# - pt (Point): Query point in the same CRS as the line.
# - road_line (LineString): Road centerline.
# Outputs:
# - float or None: Bearing in degrees modulo 180, or None for a degenerate segment.
def _bearing_at_point_on_line(pt: Point, road_line: LineString):
    """Road tangent bearing (0..180°) at the closest location to pt."""
    if road_line.is_empty or road_line.length == 0:
        return None
    d = road_line.project(pt)
    eps = max(0.5, min(2.0, road_line.length * 0.02))
    p0 = road_line.interpolate(max(d - eps, 0.0))
    p1 = road_line.interpolate(min(d + eps, road_line.length))
    dx, dy = (p1.x - p0.x, p1.y - p0.y)
    if dx == 0 and dy == 0:
        return None
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    return ang

# Purpose: Compute a unit normal vector that is perpendicular to the road line at the closest location to a point.
# Inputs:
# - pt (Point): Query point in the same CRS as the line.
# - road_line (LineString): Road centerline.
# Outputs:
# - tuple[float,float] or None: (nx, ny) components of the unit normal, or None for a degenerate segment.
def _normal_vector_at_point_on_line(pt: Point, road_line: LineString):
    """Unit normal (perpendicular) to the road at the seed point."""
    if road_line.is_empty or road_line.length == 0:
        return None
    d = road_line.project(pt)
    eps = max(0.5, min(2.0, road_line.length * 0.02))
    p0 = road_line.interpolate(max(d - eps, 0.0))
    p1 = road_line.interpolate(min(d + eps, road_line.length))
    vx, vy = (p1.x - p0.x), (p1.y - p0.y)
    L = np.hypot(vx, vy)
    if L == 0:
        return None
    return (-vy / L, vx / L)

# Purpose: Derive the orientation of a polygon’s long axis using its minimum rotated rectangle.
# Inputs:
# - poly (Polygon or MultiPolygon): Input polygon.
# Outputs:
# - float or None: Orientation of the long side in degrees modulo 180, or None if not computable.
def _poly_major_axis_deg(poly):
    """Orientation (0..180°) of polygon’s long axis using its min-rotated-rect."""
    if poly is None or poly.is_empty:
        return None
    try:
        if poly.geom_type == "MultiPolygon":
            poly = max(list(poly.geoms), key=lambda g: g.area)
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        best_len, best_ang = -1.0, None
        for i in range(4):
            x1,y1 = coords[i]; x2,y2 = coords[i+1]
            L = np.hypot(x2-x1, y2-y1)
            if L > best_len:
                best_len = L; best_ang = (np.degrees(np.arctan2(y2-y1, x2-x1)) % 180.0)
        return best_ang
    except Exception:
        return None

# Purpose: Build a crosswalk rectangle centered at a seed point and oriented perpendicular to the local road direction.
# Inputs:
# - pt (Point): Seed point on or near the road.
# - road_line (LineString): Road centerline used to compute orientation.
# - road_total_width (float): Curb-to-curb width in meters used to set the rectangle length.
# - stripe_width (float): Crosswalk stripe width in meters.
# - length_factor (float): Multiplier applied to road_total_width to size the rectangle length.
# Outputs:
# - Polygon or None: Oriented rectangle polygon, or None if the road geometry is degenerate.
def _oriented_crosswalk_from_point(pt: Point, road_line: LineString,
                                   road_total_width: float, stripe_width: float,
                                   length_factor: float = 1.2):
    if road_line.is_empty or road_line.length == 0:
        return None
    d = road_line.project(pt)
    eps = max(0.5, min(2.0, road_line.length * 0.02))
    p0 = road_line.interpolate(max(d - eps, 0.0))
    p1 = road_line.interpolate(min(d + eps, road_line.length))
    dx, dy = (p1.x - p0.x, p1.y - p0.y)
    angle = np.degrees(np.arctan2(dy, dx)) + 90.0
    L = float(road_total_width) * float(length_factor)
    W = float(stripe_width)
    rect = box(-L/2, -W/2, L/2, W/2)
    rect = rotate(rect, angle, origin=(0, 0), use_radians=False)
    rect = translate(rect, xoff=pt.x, yoff=pt.y)
    return rect

# Purpose: Test whether a rectangle overlaps pedestrian boundaries on both of its short ends.
# Inputs:
# - rect (Polygon): Candidate crosswalk rectangle.
# - sw_union (LineString/MultiLineString/Polygon): Unary union of sidewalk and pedestrian-zone boundaries in LOCAL_CRS.
# Outputs:
# - bool: True when the rectangle intersects pedestrian boundaries in at least two distinct places.
def _touches_sidewalk_both_sides(rect, sw_union) -> bool:
    if sw_union is None or sw_union.is_empty or rect is None or rect.is_empty:
        return False
    inter = rect.intersection(sw_union)
    if inter.is_empty:
        return False
    if hasattr(inter, "geoms"):
        return sum(1 for g in inter.geoms if not g.is_empty) >= 2
    return False

# Purpose: Measure curb-to-curb width at a candidate location by intersecting a normal chord with the road union.
# Inputs:
# - pt (Point): Candidate point in LOCAL_CRS.
# - road_line (LineString): Road centerline in LOCAL_CRS.
# - roads_union_poly (Polygon/MultiPolygon): Unary union of road polygons in LOCAL_CRS.
# - hw (str): OSM 'highway' value used for attribute fallback.
# - lanes (any): OSM 'lanes' value used for attribute fallback.
# - probe_len (float): Half length of the probing chord in meters.
# - corridor_scale (float): Scale factor for the fallback corridor width.
# Outputs:
# - float: Estimated curb-to-curb width in meters; falls back to an attribute-based estimate when geometry probing fails.
def _curb2curb_width_for_candidate(
    pt: Point,
    road_line: LineString,
    roads_union_poly,
    hw: str,
    lanes,
    probe_len: float = 160.0,
    corridor_scale: float = 1.6,
):
    """Measure curb-to-curb width along road normal (with fallbacks)."""
    if (roads_union_poly is None or getattr(roads_union_poly, "is_empty", True) or
        road_line is None or road_line.is_empty or pt is None or pt.is_empty):
        return _infer_total_width_from_attrs(hw, lanes)
    nv = _normal_vector_at_point_on_line(pt, road_line)
    if nv is None:
        return _infer_total_width_from_attrs(hw, lanes)
    nx, ny = nv
    a = (pt.x - nx * probe_len, pt.y - ny * probe_len)
    b = (pt.x + nx * probe_len, pt.y + ny * probe_len)
    chord = LineString([a, b])
    inter = _safe_intersection(roads_union_poly, chord, label="roads∩chord", fallback="empty")
    L = _length_from_intersection(inter, pt)
    if L > 0:
        return L
    base = _infer_total_width_from_attrs(hw, lanes)
    corr_w = max(6.0, corridor_scale * max(base, 8.0))
    corridor = _safe_buffer(chord, corr_w * 0.5, cap_style=2, join_style=2)
    region = _safe_intersection(roads_union_poly, corridor, label="roads∩corridor", fallback="a")
    inter2 = _safe_intersection(region, chord, label="(roads∩corridor)∩chord", fallback="empty")
    L2 = _length_from_intersection(inter2, pt)
    return L2 if L2 > 0 else base

# Purpose: Construct and quality-gate a crosswalk rectangle so as to connect curbs and, if present, sidewalks.
# Algorithm:
# - Build an oriented rectangle at the seed point using the road’s normal.
# - Try multiple length factors and curb overlaps from the prebuilt road buffer cache.
# - Prefer rectangles that touch sidewalks on both sides; otherwise keep the best clipped candidate.
# Inputs:
# - pt (Point): Candidate point in LOCAL_CRS.
# - road_geom (LineString): Road centerline in LOCAL_CRS.
# - span (float): Curb-to-curb distance in meters at the seed point.
# - stripe_w (float): Crosswalk width in meters.
# - sw_union (geometry): Union of sidewalks and pedestrian-zone boundaries in LOCAL_CRS.
# - road_bufs (dict[float, geometry]): Cache of road-union buffers keyed by curb overlap in meters.
# - grow_steps (Iterable[float] | None): Additional length factors to try.
# - start_overlap (float), max_overlap (float), step_overlap (float): Overlap search range in meters.
# Outputs:
# - Polygon or None: Best crosswalk rectangle in LOCAL_CRS, clipped by the road buffers.
def _make_stripe_connecting(
    pt: Point,
    road_geom: LineString,
    span: float,
    stripe_w: float,
    sw_union,
    road_bufs: dict,
    grow_steps=None,
    start_overlap=None,
    max_overlap=None,
    step_overlap=None,
):
    """
    Build oriented, quality-gated crosswalk rectangle at `pt`:
    1) create a rectangle perpendicular to the road,
    2) try several length factors,
    3) clip into prebuilt road buffers with varying curb overlap,
    4) prefer rectangles that touch sidewalks on both sides,
       fall back to the largest clipped area if none do.
    """
    if span <= 0 or road_geom is None or road_geom.is_empty:
        return None

    grow_steps    = tuple(grow_steps or CROSSWALK_CONNECT.get("grow_steps", (1.6, 2.0, 2.6)))
    start_overlap = float(start_overlap if start_overlap is not None else CROSSWALK_CONNECT.get("curb_overlap_m", 0.5))
    max_overlap   = float(max_overlap   if max_overlap   is not None else CROSSWALK_CONNECT.get("max_curb_overlap_m", start_overlap + 1.0))
    step_overlap  = float(step_overlap  if step_overlap  is not None else CROSSWALK_CONNECT.get("curb_overlap_step_m", 0.25))

    # Slight boost to the measured curb-to-curb span helps skew/diagonal roads
    span = float(span) * (1.0 + float(CROSSWALK_CONNECT.get("fallback_boost_pct", 0.10)))

    # Use cached road buffers so we don't buffer inside the loop
    try_ovs = []
    ov = start_overlap
    while ov <= max_overlap + 1e-9:
        key = round(ov, 3)
        if key in road_bufs:
            try_ovs.append(key)
        ov += step_overlap

    best = None
    for Lfac in grow_steps:
        rect = _oriented_crosswalk_from_point(pt, road_geom, span, stripe_w, length_factor=Lfac)
        if rect is None or rect.is_empty:
            continue
        if not _is_good_crosswalk_rect(rect, pt, road_geom, span, stripe_w):
            continue

        for key in try_ovs:
            buf = road_bufs.get(key)
            clipped = rect if (buf is None or getattr(buf, "is_empty", False)) else _safe_intersection(
                rect, buf, label="stripe∩roadbuf", fallback="a"
            )
            # Perfect case: touches sidewalks on both sides
            if not clipped.is_empty and _touches_sidewalk_both_sides(clipped, sw_union):
                return clipped

            # Otherwise keep the largest viable clipped rect as a fallback
            if best is None or (not clipped.is_empty and clipped.area > best.area):
                best = clipped

    return best


# -------------------- Crosswalks (points/lines/polys  to  polygons) --------------------
# Purpose: Convert OSM crosswalk points and segments to oriented crosswalk polygons using road geometry and attributes.
# Key steps in LOCAL_CRS:
# - Build unions and spatial indexes for roads and sidewalks.
# - For each crosswalk seed, select candidate road segments in a search radius.
# - Filter candidates by drivable highway, road length near the seed, and intersection degree.
# - Estimate curb-to-curb span using geometric probing with attribute fallbacks.
# - Generate oriented rectangles, clip to road buffers, and keep the highest scoring candidate.
# Inputs:
# - crosswalks_ll (GeoDataFrame): Crosswalk points and/or short line segments in WGS84.
# - roads_ll (GeoDataFrame): Cleaned road centerlines in WGS84 with 'highway' and 'lanes' columns if available.
# - sidewalks_ll (GeoDataFrame): Sidewalk and pedestrian-zone polygons in WGS84.
# - streets_poly_ll (GeoDataFrame): Road polygons in WGS84 to supply the union used for clipping.
# Outputs:
# - GeoDataFrame: Crosswalk polygons in WGS84 with column 'type' set to 'crosswalk'.
def prepare_crosswalk_polygons_dynamic(
    crosswalks_ll: gpd.GeoDataFrame,
    roads_ll: gpd.GeoDataFrame,
    sidewalks_ll: gpd.GeoDataFrame,
    streets_poly_ll: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if crosswalks_ll.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    with log_step("Prepare crosswalk polygons"):
        cw = crosswalks_ll.to_crs(LOCAL_CRS).copy()
        roads_local = roads_ll.to_crs(LOCAL_CRS).copy()
        if not roads_local.empty:
            roads_local["seg_len_m"] = roads_local.geometry.length
        t0 = time.perf_counter()
        road_union_lines = roads_local.geometry.union_all() if not roads_local.empty else None
        logger.info(f"Roads union (lines) ready in {time.perf_counter()-t0:.2f}s")
        t1 = time.perf_counter()
        roads_union_poly = (
            _fix_geom(streets_poly_ll.to_crs(LOCAL_CRS).geometry.union_all())
            if not streets_poly_ll.empty else None
        )
        logger.info(f"Roads union (polygons) ready in {time.perf_counter()-t1:.2f}s")
        t2 = time.perf_counter()
        sw_union = sidewalks_ll.to_crs(LOCAL_CRS).geometry.union_all() if not sidewalks_ll.empty else None
        logger.info(f"Sidewalks union ready in {time.perf_counter()-t2:.2f}s")
        start_ov = CROSSWALK_CONNECT.get("curb_overlap_m", 0.75)
        max_ov   = CROSSWALK_CONNECT.get("max_curb_overlap_m", 1.8)
        step_ov  = CROSSWALK_CONNECT.get("curb_overlap_step_m", 0.25)
        road_bufs = _build_road_buffer_cache(roads_union_poly, start_ov, max_ov, step_ov)
        curb_eps   = float(CROSSWALK_CONNECT.get("curb_overlap_m", 0.35))
        curb_union = _safe_buffer(roads_union_poly, curb_eps, cap_style=2, join_style=2) if roads_union_poly else None

        polygons, missed = [], []
        SNAP_TOL = float(CROSSWALK_RULES["snap_tolerance_m"])
        DEG_R    = float(CROSSWALK_RULES["intersection_radius_m"])
        MIN_DEG  = int(CROSSWALK_RULES["min_deg_for_point"])
        MIN_SEG_L= float(CROSSWALK_RULES["min_nearest_road_len_m"])
        grow_seq = tuple(CROSSWALK_CONNECT.get("grow_steps", (1.6, 2.0, 2.6)))

        # POINTS
        cw_pts = cw[cw.geom_type == "Point"].copy()
        n_pts = len(cw_pts)
        logger.info(f"Crosswalk seeds: {n_pts} points")
        start = time.perf_counter()
        last  = start
        report_every = max(50, n_pts // 20) if n_pts else 50

        for i, row in cw_pts.iterrows():
            pt = row.geometry
            is_marked = str(row.get("crossing", "")).lower() == "marked"
            if pt is None or pt.is_empty:
                continue
            width = float(row.get("crossing:width", BUFFERS["crosswalk_width"]))
            candidates = _nearby_roads(pt, roads_local, radius_m=22.0, maxn=6)
            if not candidates and road_union_lines is not None and not road_union_lines.is_empty:
                nearest_pt = nearest_points(pt, road_union_lines)[1]
                candidates = [type("T", (), {"Index": -1, "geometry": LineString([nearest_pt, translate(nearest_pt, xoff=0.5, yoff=0)]),
                                             "highway":"", "lanes":np.nan, "seg_len_m":np.nan})()]
            best_rect = None
            best_score = (-1, -1e9, 0.0)
            for cand in candidates:
                road_geom = cand.geometry
                hw   = (getattr(cand, "highway", "") or "").lower()
                segL = getattr(cand, "seg_len_m", np.nan)
                deg  = _local_road_degree(pt, roads_local, DEG_R)
                drivable_ok = (_is_drivable_highway(hw) if hw else True) or (is_marked and (hw == "service"))
                midblock_ok = (is_marked and deg == 1 and near_ok)
                stub_fail   = (not np.isnan(segL) and segL < MIN_SEG_L and deg < MIN_DEG)
                near_ok     = True if (road_union_lines is None or road_union_lines.is_empty) else (pt.distance(road_union_lines) <= SNAP_TOL)
                if (road_geom is None or road_geom.is_empty or not near_ok or not drivable_ok):
                    continue

                # Don’t drop short “stubs” if this is a marked crossing and we’re mid-block
                if stub_fail and not midblock_ok:
                    continue

                # Keep requiring ≥2 legs unless it’s a marked mid-block case
                if deg < MIN_DEG and not midblock_ok:
                    continue
                ln = getattr(cand, "lanes", np.nan)
                attr_w = _infer_total_width_from_attrs(hw, ln)
                span = _curb2curb_width_for_candidate(
                    pt, road_geom, roads_union_poly, hw, ln,
                    probe_len=160.0, corridor_scale=1.6
                )
                rect = _make_stripe_connecting(
                    pt, road_geom, span, width, sw_union, road_bufs,
                    grow_steps=grow_seq, start_overlap=start_ov, max_overlap=max_ov, step_overlap=step_ov
                )
                sc = _score_stripe(rect, roads_union_poly, sw_union, span=span, attr_w=attr_w)
                if sc > best_score:
                    best_score = sc
                    best_rect = rect

            if best_rect is not None and not best_rect.is_empty:
                stripe = best_rect if (curb_union is None or curb_union.is_empty) else best_rect.intersection(curb_union)
                accept = False
                if not stripe.is_empty:
                    if _touches_sidewalk_both_sides(stripe, sw_union):
                        accept = True
                    elif CROSSWALK_FALLBACKS["enable_if_no_sidewalks"]:
                        ok_curbs = CROSSWALK_FALLBACKS["accept_if_two_curb_hits"] and _touches_road_both_curbs(stripe, roads_union_poly)
                        ok_area  = (roads_union_poly is not None and not roads_union_poly.is_empty
                                    and stripe.intersection(roads_union_poly).area >= CROSSWALK_FALLBACKS["min_road_overlap_m2"])
                        near_rd  = (road_union_lines is None or road_union_lines.is_empty or
                                    row.geometry.distance(road_union_lines) <= CROSSWALK_FALLBACKS["max_center_to_road_m"])
                        accept = ok_curbs or (ok_area and near_rd)
                if accept:
                    polygons.append(stripe)
                else:
                    missed.append(i)
            else:
                missed.append(i)

            if n_pts and (((len(polygons)+len(missed)) % report_every == 0) or (time.perf_counter()-last > 5.0)):
                done = len(polygons) + len(missed)
                pct = 100.0 * done / n_pts
                logger.info(f"  … crosswalk seeds processed {done}/{n_pts} ({pct:.1f}%) in {time.perf_counter()-start:.1f}s")
                last = time.perf_counter()

        # LINES to buffer
        cw_lines = cw[cw.geom_type.isin(["LineString","MultiLineString"])].copy()
        logger.info(f"Crosswalk line/multiline features: {len(cw_lines)}")
        for i, row in cw_lines.iterrows():
            try:
                w = float(row.get("crossing:width", BUFFERS["crosswalk_width"]))
                polygons.append(row.geometry.buffer(w/2.0, cap_style=2, join_style=2))
            except Exception:
                missed.append(i)

        # POLYGONS to clean
        cw_polys = cw[cw.geom_type.isin(["Polygon","MultiPolygon"])].copy()
        logger.info(f"Crosswalk polygon/multipolygon features: {len(cw_polys)}")
        for i, row in cw_polys.iterrows():
            try:
                polygons.append(row.geometry.buffer(0))
            except Exception:
                missed.append(i)

        if missed:
            try:
                cw.loc[missed].to_file("debug_missed_crosswalks.geojson", driver="GeoJSON")
                logger.info(f"Wrote {len(missed)} missed crosswalk seeds to debug_missed_crosswalks.geojson")
            except Exception:
                pass
        logger.info(f"Crosswalks created: {len(polygons)} (missed {len(missed)})")

    out = gpd.GeoDataFrame(geometry=polygons, crs=LOCAL_CRS)
    out["type"] = "crosswalk"
    out = clean_geoms(out.to_crs(WGS84), 0.01, LOCAL_CRS)
    out = remove_small_polygons(out, min_area_m2=0.35, crs_local=LOCAL_CRS)
    return out

# Purpose: Add short sidewalk connector polygons from crosswalk short-edge midpoints to the nearest pedestrian boundary.
# Inputs:
# - crosswalk_poly_ll (GeoDataFrame): Crosswalk rectangles in WGS84.
# - sidewalks_ll (GeoDataFrame): Sidewalk polygons in WGS84.
# - ped_zones_ll (GeoDataFrame): Pedestrian-only zones in WGS84.
# Outputs:
# - GeoDataFrame: Connector polygons in WGS84 with column 'type' set to 'sidewalk'.
def add_crosswalk_connectors(crosswalk_poly_ll, sidewalks_ll, ped_zones_ll):
    if crosswalk_poly_ll.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    cw = crosswalk_poly_ll.to_crs(LOCAL_CRS).copy()
    sw = sidewalks_ll.to_crs(LOCAL_CRS) if not sidewalks_ll.empty else gpd.GeoDataFrame(geometry=[], crs=LOCAL_CRS)
    ped = ped_zones_ll.to_crs(LOCAL_CRS) if (ped_zones_ll is not None and not ped_zones_ll.empty) else gpd.GeoDataFrame(geometry=[], crs=LOCAL_CRS)
    targets = []
    if not sw.empty:
        targets.extend(list(sw.geometry.boundary.values))
    if not ped.empty:
        targets.extend(list(ped.geometry.boundary.values))
    if not targets:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    tree = STRtree(targets)
    R = float(CROSSWALK_FALLBACKS["connector_search_m"])
    W = float(CROSSWALK_FALLBACKS["connector_width_m"])
    swu = sw.geometry.union_all() if not sw.empty else None
    conns = []
    for g in cw.geometry:
        if g is None or g.is_empty:
            continue
        if swu is not None and g.intersects(swu):
            continue  # already touching sidewalks
        for mid in _short_edge_midpoints(g):
            cands = tree.query(mid.buffer(R))
            if not cands:
                continue
            nearest = min(cands, key=lambda geom: mid.distance(geom))
            p2 = nearest_points(mid, nearest)[1]
            line = LineString([mid.coords[0], (p2.x, p2.y)])
            buf = line.buffer(W/2.0, cap_style=2, join_style=2)
            if buf is not None and not buf.is_empty:
                conns.append(buf)
    out = gpd.GeoDataFrame(geometry=conns, crs=LOCAL_CRS).to_crs(WGS84)
    if not out.empty:
        out["type"] = "sidewalk"  # treat as sidewalk stubs
    return out

# -------------------- Crosswalk subtraction from streets --------------------
# Purpose: Remove crosswalk rectangles from street polygons to avoid double counting and to leave road-asphalt only.
# Inputs:
# - roads_poly_ll (GeoDataFrame): Street polygons in WGS84.
# - crosswalk_poly_ll (GeoDataFrame): Crosswalk polygons in WGS84.
# - roads_lines_ll (GeoDataFrame | None): Optional road centerlines in WGS84 used to filter for near-perpendicular cuts.
# Outputs:
# - GeoDataFrame: Street polygons in WGS84 with crosswalk areas subtracted; column 'type' is ensured to be 'street'.
def subtract_crosswalks_from_roads_localized(
    roads_poly_ll: gpd.GeoDataFrame,
    crosswalk_poly_ll: gpd.GeoDataFrame,
    roads_lines_ll: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Subtract crosswalks from road polygons, keeping only near-perpendicular pieces."""
    if roads_poly_ll.empty or crosswalk_poly_ll.empty:
        out = roads_poly_ll.copy()
        if "type" not in out.columns:
            out = out.assign(type="street")
        return out

    with log_step("Subtract crosswalks from street polygons (perpendicular-only)"):
        roads_l = roads_poly_ll.to_crs(LOCAL_CRS).copy()
        cws_l   = crosswalk_poly_ll.to_crs(LOCAL_CRS).copy()
        cws_l["geometry"] = cws_l.buffer(-0.25, cap_style=2, join_style=2)
        cws_l = cws_l[~cws_l.geometry.is_empty]

        road_tree = None
        if roads_lines_ll is not None and not roads_lines_ll.empty:
            roads_lines_l = roads_lines_ll.to_crs(LOCAL_CRS)
            line_geoms = list(roads_lines_l.geometry.values)
            if line_geoms:
                road_tree = STRtree(line_geoms)

        def _best_road_for_poly(poly):
            if road_tree is None or poly is None or poly.is_empty:
                return None
            roi = _safe_buffer(poly, 6.0, cap_style=2, join_style=2)
            if roi is None or roi.is_empty:
                cen = poly.representative_point()
                return road_tree.nearest(cen)
            cand = road_tree.query(roi, predicate="intersects")
            if not cand:
                cen = poly.representative_point()
                return road_tree.nearest(cen)
            best_g, best_len = None, -1.0
            for idx in cand:
                g = line_geoms[idx]
                try:
                    inter = g.intersection(roi)
                    L = inter.length if not inter.is_empty else 0.0
                except Exception:
                    L = 0.0
                if L > best_len:
                    best_len = L; best_g = g
            return best_g

        kept, skipped_not_perp = [], 0
        PERP_TOL_DEG = CROSSWALK_SHAPE_LIMITS["max_axis_misalignment_deg"]  # e.g. 25°

        if road_tree is None:
            kept = list(cws_l.geometry.values)
        else:
            for g in cws_l.geometry:
                try:
                    cw_ang = _poly_major_axis_deg(g)
                    if cw_ang is None:
                        kept.append(g); continue
                    base_line = _best_road_for_poly(g)
                    if base_line is None:
                        kept.append(g); continue
                    bear = _bearing_at_point_on_line(g.representative_point(), base_line)
                    if bear is None:
                        kept.append(g); continue
                    normal_deg = (bear + 90.0) % 180.0
                    # keep only near-perpendicular crosswalks
                    if _acute_diff_deg(cw_ang, normal_deg) > PERP_TOL_DEG:
                        skipped_not_perp += 1
                        continue
                    kept.append(g)
                except Exception:
                    kept.append(g)

        if not kept:
            out = roads_l.to_crs(WGS84)
            if "type" not in out.columns: out = out.assign(type="street")
            return out

        logger.info(f"Crosswalk pieces kept for cutting: {len(kept)}  (skipped not-perpendicular: {skipped_not_perp})")

        tree  = STRtree(kept)
        total = len(roads_l)
        logger.info(f"Cutting {total} road polygons by {len(kept)} crosswalk pieces…")
        start = time.perf_counter()
        last  = start
        report_every = max(50, total // 20)

        new_roads = []
        for i, g in enumerate(roads_l.geometry):
            if g is None or g.is_empty:
                new_roads.append(g); continue
            cand_idxs = tree.query(g, predicate="intersects")
            if len(cand_idxs) == 0:
                new_roads.append(g); continue
            cut = g
            for idx in cand_idxs:
                cw = kept[idx]
                if cut.is_empty: break
                try:
                    if cut.intersects(cw):
                        cut = cut.difference(cw)
                except Exception:
                    try:
                        cut = _fix_geom(cut).difference(_fix_geom(cw))
                    except Exception:
                        pass
            new_roads.append(cut)

            if ((i + 1) % report_every == 0) or (time.perf_counter()-last > 5.0):
                pct = 100.0 * (i + 1) / total
                logger.info(f"  … cut {i+1}/{total} ({pct:.1f}%) in {time.perf_counter()-start:.1f}s")
                last = time.perf_counter()

        roads_l["geometry"] = new_roads
        out = roads_l.to_crs(WGS84)
        if "type" not in out.columns: out = out.assign(type="street")
        out = clean_geoms(out, 0.02, LOCAL_CRS)
        return out

# -------------------- QA metrics & flags --------------------
# Purpose: Compute basic metrics for the processed layers.
# Inputs:
# - streets_poly_ll (GeoDataFrame): Street polygons in WGS84.
# - sidewalks_out (GeoDataFrame): Sidewalk and connectors polygons in WGS84.
# - crosswalk_poly_ll (GeoDataFrame): Crosswalk polygons in WGS84.
# Outputs:
# - dict: Summary including total areas in square kilometers and feature counts per layer.
def summarize_layers(streets_poly_ll, sidewalks_out, crosswalk_poly_ll):
    def _area_km2(gdf):
        if gdf.empty: return 0.0
        return gdf.to_crs(LOCAL_CRS).area.sum() / 1e6
    metrics = {
        "streets_polygons_km2": round(_area_km2(streets_poly_ll), 4),
        "sidewalks_polygons_km2": round(_area_km2(sidewalks_out), 4),
        "crosswalks_polygons_km2": round(_area_km2(crosswalk_poly_ll), 6),
        "counts": {
            "streets": int(len(streets_poly_ll)),
            "sidewalks": int(len(sidewalks_out)),
            "crosswalks": int(len(crosswalk_poly_ll)),
        }
    }
    logger.info(f"Metrics: {metrics}")
    return metrics

# Purpose: Flag crosswalk polygons that are far from any road or that do not touch sidewalks.
# Inputs:
# - crosswalk_poly_ll (GeoDataFrame): Crosswalk polygons in WGS84.
# - roads_ll (GeoDataFrame): Road centerlines in WGS84.
# - sidewalks_ll (GeoDataFrame): Sidewalk polygons in WGS84.
# Outputs:
# - GeoDataFrame: Crosswalk polygons in WGS84 with boolean columns 'flag_far_from_road' and 'flag_not_touching_sidewalk'.
def flag_crosswalk_anomalies(crosswalk_poly_ll, roads_ll, sidewalks_ll):
    """Flag crosswalks far from roads or not touching sidewalks."""
    if crosswalk_poly_ll.empty:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    cws = crosswalk_poly_ll.to_crs(LOCAL_CRS).copy()
    roads_l = roads_ll.to_crs(LOCAL_CRS) if not roads_ll.empty else gpd.GeoDataFrame(geometry=[], crs=LOCAL_CRS)
    swu = sidewalks_ll.to_crs(LOCAL_CRS).geometry.union_all() if not sidewalks_ll.empty else None
    ru  = roads_l.geometry.union_all() if not roads_l.empty else None
    far, no_sw = [], []
    for idx, g in cws.geometry.items():
        if ru is not None and g.distance(ru) > 8.0: far.append(idx)
        if swu is not None and not g.intersects(swu): no_sw.append(idx)
    cws["flag_far_from_road"] = cws.index.isin(far)
    cws["flag_not_touching_sidewalk"] = cws.index.isin(no_sw)
    return cws.to_crs(WGS84)

# -------------------- Main Processing --------------------
# Purpose: End-to-end pipeline to read raw layers, normalize and clean linework and polygons, infer widths,
#          generate crosswalk polygons, subtract crosswalks from streets, and write processed outputs.
# Inputs:
# - None. Reads files from the project's 'outputs' directory:
#   raw_roads.geojson, raw_sidewalks.geojson, raw_crosswalks.geojson,
#   raw_curb_ramps.geojson, raw_buildings.geojson, raw_pedestrian_zones.geojson
# Outputs (written to 'outputs' directory):
# - processed_streets_polygons.geojson: Road-asphalt polygons with type='street' (WGS84).
# - processed_sidewalks_polygons.geojson: Sidewalk and connector polygons with type='sidewalk' (WGS84).
# - processed_crosswalks_polygons.geojson: Crosswalk rectangles with type='crosswalk' (WGS84).
# - processed_streets_buildings.geojson: Road polygons with building footprints (WGS84).
# - processed_pedestrian_network.geojson: Combined pedestrian features (sidewalks, crosswalks, curb ramps, ped zones) (WGS84).
# - metrics.json: Summary metrics for the above layers.
def process_data():
    logger.info("Starting processing pipeline…")
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    roads_ll       = safe_read(outputs_dir / "raw_roads.geojson")
    sidewalks_ll   = safe_read(outputs_dir / "raw_sidewalks.geojson")
    crosswalks_ll  = safe_read(outputs_dir / "raw_crosswalks.geojson")
    curb_ramps_ll  = safe_read(outputs_dir / "raw_curb_ramps.geojson")
    buildings_ll   = safe_read(outputs_dir / "raw_buildings.geojson")
    ped_zones_ll   = safe_read(outputs_dir / "raw_pedestrian_zones.geojson")

    with log_step("Fix street geometries"):
        roads_ll_fixed = fix_street_geometries(roads_ll, tolerance=1.5)

    with log_step("Buffer roads into polygons"):
        streets_poly_ll = buffer_roads_hierarchical(roads_ll_fixed, buildings_ll)

    with log_step("Clean buffered streets"):
        streets_poly_ll = clean_geoms(streets_poly_ll, 0.02, LOCAL_CRS)
        streets_poly_ll = remove_small_polygons(streets_poly_ll, min_area_m2=0.5, crs_local=LOCAL_CRS)

    with log_step("Generate crosswalk polygons"):
        crosswalk_poly_ll = prepare_crosswalk_polygons_dynamic(
            crosswalks_ll, roads_ll_fixed, sidewalks_ll, streets_poly_ll
        )

    with log_step("Subtract crosswalks from streets"):
        streets_cut_ll = subtract_crosswalks_from_roads_localized(streets_poly_ll, crosswalk_poly_ll, roads_ll_fixed)

    # Sidewalks
    with log_step("Build sidewalk polygons"):
        sidewalks_out = gpd.GeoDataFrame(geometry=[], crs=WGS84)
        if not sidewalks_ll.empty:
            sw_local = sidewalks_ll.to_crs(LOCAL_CRS).copy()
            sw_local["geometry"] = sw_local.buffer(float(BUFFERS["sidewalk_width"]) / 2.0, cap_style=2, join_style=2)
            sidewalks_out = sw_local.to_crs(WGS84).assign(type="sidewalk")
            sidewalks_out = close_sidewalk_gaps(sidewalks_out, gap_m=0.18)
            sidewalks_out = remove_small_polygons(sidewalks_out, min_area_m2=0.25, crs_local=LOCAL_CRS)

    # Pedestrian zones (needed by connectors)
    with log_step("Prep pedestrian zones"):
        ped_out = ped_zones_ll.copy() if not ped_zones_ll.empty else gpd.GeoDataFrame(geometry=[], crs=WGS84)
        if not ped_out.empty and "type" not in ped_out.columns:
            ped_out = ped_out.assign(type="pedestrian_zone")

    # Build crosswalk  to  sidewalk connectors AFTER ped_out exists
    with log_step("Build crosswalk→sidewalk connectors (fallback)"):
        try:
            connectors = add_crosswalk_connectors(crosswalk_poly_ll, sidewalks_out, ped_out)
            if not connectors.empty:
                sidewalks_out = gpd.GeoDataFrame(
                    pd.concat([sidewalks_out, connectors], ignore_index=True), crs=WGS84
                )
                logger.info(f"Added {len(connectors)} sidewalk connectors.")
        except Exception as e:
            logger.warning(f"Connector build skipped: {e}")

    with log_step("Trim sidewalks to curb (difference vs streets)"):
        if not sidewalks_out.empty and not streets_poly_ll.empty:
            st_union_l = _fix_geom(streets_poly_ll.to_crs(LOCAL_CRS).geometry.union_all())
            sw_l = sidewalks_out.to_crs(LOCAL_CRS).copy()
            curb_clear = 0.05  # 5 cm margin to avoid slivers
            st_buf = _safe_buffer(st_union_l, curb_clear, cap_style=2, join_style=2)
            sw_l["geometry"] = sw_l.geometry.difference(st_buf)
            sidewalks_out = clean_geoms(sw_l.to_crs(WGS84), 0.01, LOCAL_CRS)

    # Curb ramps
    with log_step("Build curb ramp buffers"):
        curb_out = gpd.GeoDataFrame(geometry=[], crs=WGS84)
        if not curb_ramps_ll.empty:
            cr_local = curb_ramps_ll.to_crs(LOCAL_CRS).copy()
            cr_local["geometry"] = cr_local.buffer(0.75, cap_style=1, join_style=1)
            curb_out = cr_local.to_crs(WGS84)
            curb_out["type"] = "curb_ramp"

    # --- Enforce Polygon-only for all outgoing layers ---
    with log_step("Enforce Polygon-only geometry for outputs"):
        streets_cut_ll     = ensure_polygons_only(streets_cut_ll,     log_label="final streets")
        buildings_tagged   = buildings_ll.copy()
        buildings_tagged["type"] = "house"
        buildings_tagged   = ensure_polygons_only(buildings_tagged,   log_label="buildings")
        sidewalks_out      = ensure_polygons_only(sidewalks_out,      log_label="sidewalks")
        crosswalk_poly_ll  = ensure_polygons_only(crosswalk_poly_ll,  log_label="crosswalks")
        ped_out            = ensure_polygons_only(ped_out,            log_label="pedestrian_zones")
        curb_out           = ensure_polygons_only(curb_out,           log_label="curb_ramps")

    # Save streets + buildings
    with log_step("Write processed_streets_buildings.geojson"):
        streets_for_concat   = ensure_geom_type(streets_cut_ll,   "street", crs=WGS84)
        buildings_for_concat = ensure_geom_type(buildings_tagged, "house",  crs=WGS84)
        streets_buildings = gpd.GeoDataFrame(
            pd.concat([streets_for_concat, buildings_for_concat], ignore_index=True),
            crs=WGS84
        )
        streets_buildings.to_file(outputs_dir / "processed_streets_buildings.geojson", driver="GeoJSON")

    # Pedestrian network
    with log_step("Write processed_pedestrian_network.geojson"):
        ped_parts = [df for df in [
            sidewalks_out[["geometry","type"]] if not sidewalks_out.empty else None,
            crosswalk_poly_ll[["geometry","type"]] if not crosswalk_poly_ll.empty else None,
            curb_out[["geometry","type"]] if not curb_out.empty else None,
            ped_out[["geometry","type"]] if not ped_out.empty else None
        ] if df is not None]
        if ped_parts:
            pedestrian_features = gpd.GeoDataFrame(pd.concat(ped_parts, ignore_index=True), crs=WGS84)
            pedestrian_features.to_file(outputs_dir / "processed_pedestrian_network.geojson", driver="GeoJSON")

    # Accessibility
    with log_step("Write processed_accessibility.geojson"):
        acc_parts = [df for df in [
            curb_out[["geometry","type"]] if not curb_out.empty else None,
            crosswalk_poly_ll[["geometry","type"]] if not crosswalk_poly_ll.empty else None
        ] if df is not None]
        if acc_parts:
            accessibility = gpd.GeoDataFrame(pd.concat(acc_parts, ignore_index=True), crs=WGS84)
            accessibility.to_file(outputs_dir / "processed_accessibility.geojson", driver="GeoJSON")

    # QA: metrics + anomaly flags
    with log_step("Compute QA metrics and flags"):
        _ = summarize_layers(streets_cut_ll, sidewalks_out, crosswalk_poly_ll)
        flags = flag_crosswalk_anomalies(crosswalk_poly_ll, roads_ll_fixed, sidewalks_ll)
        if not flags.empty:
            flags.to_file(outputs_dir / "debug_crosswalk_flags.geojson", driver="GeoJSON")

    logger.info("Processing pipeline complete.")

if __name__ == "__main__":
    process_data()