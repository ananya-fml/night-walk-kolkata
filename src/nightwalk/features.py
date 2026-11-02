"""Segment-level features from OSM layers and night-time lights.

For every street segment:

  lamps_per_100m   mapped street lamps within LAMP_RADIUS_M
  lit_tag          OSM lit=* (yes / no / blank)
  ntl              VIIRS radiance at the segment midpoint (area context)
  open_expected    expected number of places open at NIGHT_TIME within
                   POI_RADIUS_M (observed opening_hours, else a category prior)
  open_observed    places within reach whose hours were actually parsed
  open_imputed     places whose open status came from the prior
  pois             all street-facing places within reach (daytime frontage)
  isolation        share of the segment within ISOLATION_RADIUS_M of parks,
                   rail land, water, industrial or vacant land
  guardian         police station or hospital within GUARDIAN_RADIUS_M
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import polygonize, unary_union

from .config import (ALWAYS_OPEN, GUARDIAN_RADIUS_M, HOUR_FACTOR,
                     ISOLATING_LANDUSE, ISOLATION_RADIUS_M,
                     LAMP_RADIUS_M, NIGHT_TIME, OPEN_PRIOR,
                     OPEN_PRIOR_DEFAULT, OPEN_PRIOR_DEFAULT_SHOP,
                     POI_RADIUS_M, UTM_EPSG)

# Amenities that are street furniture rather than places with people in them
FURNITURE = {
    "bench", "waste_basket", "parking", "parking_space", "bicycle_parking",
    "motorcycle_parking", "post_box", "drinking_water", "vending_machine",
    "recycling", "waste_disposal", "shelter", "toilets", "telephone", "clock",
    "fountain", "water_point", "parking_entrance", "charging_station",
    "letter_box", "grit_bin", "hunting_stand", "loading_dock",
}
GUARDIANS = {"police", "hospital"}


# ── POIs and opening hours ─────────────────────────────────────────────
def _coords(e):
    if "lat" in e:
        return e["lon"], e["lat"]
    if "center" in e:
        return e["center"]["lon"], e["center"]["lat"]
    return None


def category(tags):
    if "amenity" in tags:
        return tags["amenity"]
    if "shop" in tags:
        return "shop:" + tags["shop"]
    if "tourism" in tags:
        return "hotel" if tags["tourism"] in {"hotel", "guest_house", "hostel"} else tags["tourism"]
    return tags.get("leisure", "other")


def open_status(tags, when=NIGHT_TIME):
    """Return (probability open, observed?) for one place at `when`."""
    import opening_hours as oh
    raw = tags.get("opening_hours", "").strip()
    if raw:
        try:
            return float(oh.OpeningHours(raw).is_open(when)), True
        except Exception:
            pass                      # unparseable: fall through to the prior
    cat = category(tags)
    key = cat.split(":", 1)[-1]
    if key in OPEN_PRIOR:
        p = OPEN_PRIOR[key]
    elif cat.startswith("shop:"):
        p = OPEN_PRIOR_DEFAULT_SHOP
    else:
        p = OPEN_PRIOR_DEFAULT
    if key not in ALWAYS_OPEN:
        p = min(1.0, p * hour_factor(when))
    return p, False


def hour_factor(when):
    return HOUR_FACTOR.get(when.hour, 1.0)


def pois_frame(pois, when=NIGHT_TIME):
    rows = []
    for e in pois["elements"]:
        t = e.get("tags", {})
        if t.get("amenity") in FURNITURE:
            continue
        xy = _coords(e)
        if xy is None:
            continue
        p, obs = open_status(t, when)
        rows.append({"osm": f"{e['type'][0]}{e['id']}", "category": category(t),
                     "name": t.get("name", ""), "p_open": p, "observed": obs,
                     "has_hours": bool(t.get("opening_hours")),
                     "guardian": t.get("amenity") in GUARDIANS,
                     "geometry": Point(xy)})
    if not rows:
        return gpd.GeoDataFrame(columns=["p_open", "observed", "guardian", "geometry"],
                                geometry="geometry", crs=4326).to_crs(UTM_EPSG)
    return gpd.GeoDataFrame(rows, crs=4326).to_crs(UTM_EPSG)


def lamps_frame(lamps):
    pts = [Point(e["lon"], e["lat"]) for e in lamps["elements"] if e["type"] == "node"]
    return gpd.GeoDataFrame(geometry=pts, crs=4326).to_crs(UTM_EPSG)


# ── Land use ────────────────────────────────────────────────────────────
def _is_isolating(tags):
    return any(tags.get(k) == v for k, v in ISOLATING_LANDUSE)


def landuse_frame(landuse):
    polys = []
    for e in landuse["elements"]:
        if not _is_isolating(e.get("tags", {})):
            continue
        if e["type"] == "way" and "geometry" in e:
            ring = [(p["lon"], p["lat"]) for p in e["geometry"]]
            if len(ring) >= 4 and ring[0] == ring[-1]:
                polys.append(Polygon(ring))
        elif e["type"] == "relation":
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                     for m in e.get("members", [])
                     if m.get("role") == "outer" and len(m.get("geometry", [])) >= 2]
            polys.extend(polygonize(lines))
    g = gpd.GeoDataFrame(geometry=[p.buffer(0) for p in polys], crs=4326)
    return g[~g.is_empty].to_crs(UTM_EPSG)


# ── Joins ───────────────────────────────────────────────────────────────
def _pairs(edges, points, dist):
    """(edge index, point index) pairs within `dist` metres."""
    if len(points) == 0:
        return np.empty(0, int), np.empty(0, int)
    pi, ei = edges.sindex.query(points.geometry, predicate="dwithin", distance=dist)
    return ei, pi


def add_features(edges, pois, lamps, landuse, ntl_path=None):
    e = edges.copy()
    n = len(e)

    # lamps
    ei, _ = _pairs(e, lamps, LAMP_RADIUS_M)
    cnt = np.bincount(ei, minlength=n)
    e["lamps"] = cnt
    e["lamps_per_100m"] = 100 * cnt / e.length_m.clip(lower=10)

    # places
    ei, pi = _pairs(e, pois, POI_RADIUS_M)
    p = pois.iloc[pi]
    def agg(vals):
        return np.bincount(ei, weights=np.asarray(vals, float), minlength=n)
    e["pois"] = np.bincount(ei, minlength=n)
    e["open_expected"] = agg(p.p_open)
    e["open_observed"] = agg(p.observed)
    e["open_imputed"] = e.pois - e.open_observed

    # guardians
    g = pois[pois.guardian]
    ei, _ = _pairs(e, g, GUARDIAN_RADIUS_M)
    e["guardian"] = np.bincount(ei, minlength=n) > 0

    # isolation: share of segment length near isolating land
    if len(landuse):
        zone = unary_union(landuse.geometry.buffer(ISOLATION_RADIUS_M))
        zone_g = gpd.GeoDataFrame(geometry=[zone], crs=e.crs)
        hit = e.sindex.query(zone_g.geometry.iloc[0], predicate="intersects")
        iso = np.zeros(n)
        if len(hit):
            sub = e.geometry.iloc[hit]
            iso[hit] = sub.intersection(zone).length / e.length_m.iloc[hit].values
        e["isolation"] = np.clip(iso, 0, 1)
    else:
        e["isolation"] = 0.0

    e["ntl"] = sample_raster(e, ntl_path) if ntl_path else np.nan
    return e


def open_at(edges, pois_raw, when):
    """open_expected / open_observed for every segment at another hour."""
    pois = pois_frame(pois_raw, when)
    ei, pi = _pairs(edges, pois, POI_RADIUS_M)
    p = pois.iloc[pi]
    n = len(edges)
    return (np.bincount(ei, weights=p.p_open.values.astype(float), minlength=n),
            np.bincount(ei, weights=p.observed.values.astype(float), minlength=n))


def sample_raster(edges, path, zero_is_missing=True):
    """VIIRS radiance at each segment midpoint.

    Rasters clipped to an administrative boundary are often zero-filled
    outside it, and no lit urban street reads exactly 0 nW/cm2/sr, so exact
    zeros are treated as missing by default. Points off the raster are
    missing too.
    """
    import rasterio
    from pyproj import Transformer
    mid = edges.geometry.interpolate(0.5, normalized=True)
    with rasterio.open(path) as src:
        tf = Transformer.from_crs(edges.crs, src.crs, always_xy=True)
        xs, ys = tf.transform(mid.x.values, mid.y.values)
        b = src.bounds
        inside = (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top)
        vals = np.full(len(xs), np.nan)
        if inside.any():
            got = np.array([v[0] for v in src.sample(zip(xs[inside], ys[inside]))], float)
            vals[inside] = got
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    if zero_is_missing:
        vals[vals == 0] = np.nan
    return vals


def coverage(edges, pois):
    """How complete are the inputs? The basis of the confidence layer."""
    return pd.Series({
        "segments with lit=* tag (%)": 100 * (edges.lit != "").mean(),
        "segments with a mapped lamp (%)": 100 * (edges.lamps > 0).mean(),
        "places with opening_hours (%)": 100 * pois.has_hours.mean() if len(pois) else np.nan,
        "places with parseable hours (%)": 100 * pois.observed.mean() if len(pois) else np.nan,
        "segments with any place in 50 m (%)": 100 * (edges.pois > 0).mean(),
    }).round(1)
