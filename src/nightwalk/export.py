"""Compact JSON bundle for the browser map.

The browser runs its own Dijkstra, so it needs the graph, not only the
geometry. Layout (all arrays, index-aligned):

  nodes   [[lon, lat], ...]                     5 decimal places (~1 m)
  edges   [[u, v, length_m, score, conf, L, A, C, shape_index], ...]
          scores are integers 0-100
  shapes  [[lon, lat, lon, lat, ...], ...]       interior vertices only
  names   street names (edge row index into it)
"""
import json
from datetime import datetime

import numpy as np

from .config import (EXPOSURE_BETA, LOW_SCORE, METRO_DWELL_S,
                     METRO_SERVICE_END, METRO_SERVICE_START, METRO_SPEED_MPS, METRO_WAIT_S,
                     NIGHT_TIME, STATION_ACCESS_M, TRANSFER_S, WALK_SPEED_MPS,
                     WEIGHTS)
from .network import node_table
from .route import metro_running


KINDS = ["street", "metro", "rail", "place", "landmark", "area"]
LANDMARK = {"hospital", "college", "university", "cinema", "police", "hotel",
            "marketplace", "place_of_worship", "bus_station", "townhall",
            "courthouse", "library", "museum", "attraction", "theatre"}


def gazetteer(edges, pois=None, stations=None, transit_raw=None, places_raw=None,
              cell_m=2000):
    """Searchable names for the map: streets, places, metro and rail stations.

    A street name used in several parts of the city keeps one entry per 2 km
    cell, at the midpoint of its longest segment there.
    """
    from .config import UTM_EPSG
    import geopandas as gpd
    items, seen = [], set()

    def add(name, lon, lat, kind):
        name = " ".join(str(name).split())
        if not name or len(name) > 80:
            return
        # stations: one entry per name (interchanges have one node per line)
        k = ((name.lower(),) if kind in ("metro", "rail")
             else (name.lower(), round(lon, 3), round(lat, 3)))
        if k in seen:
            return
        seen.add(k)
        items.append([name, round(float(lon), 5), round(float(lat), 5), KINDS.index(kind)])

    if stations is not None:
        for _, r in stations.iterrows():
            nm = r["name"].split(" (Line")[0]
            add(nm + " metro", r.lon, r.lat, "metro")
    if transit_raw is not None:
        for el in transit_raw["elements"]:
            t = el.get("tags", {})
            if el["type"] == "node" and t.get("railway") == "station" and \
                    t.get("station") not in {"subway", "light_rail"} and t.get("name"):
                add(t.get("name:en") or t["name"], el["lon"], el["lat"], "rail")
    if places_raw is not None:
        for el in places_raw["elements"]:
            t = el.get("tags", {})
            lon = el.get("lon", el.get("center", {}).get("lon"))
            lat = el.get("lat", el.get("center", {}).get("lat"))
            if lon is None or not t.get("name"):
                continue
            kind = "area" if "place" in t else "landmark"
            add(t.get("name:en") or t["name"], lon, lat, kind)
    if pois is not None and len(pois):
        ll = pois.to_crs(4326)
        for nm, cat, g in zip(pois.name, pois.category, ll.geometry):
            if nm:
                add(nm, g.x, g.y, "landmark" if cat.split(":")[-1] in LANDMARK else "place")
    named = edges[edges.name != ""]
    if len(named):
        mid = named.geometry.interpolate(0.5, normalized=True)
        df = gpd.GeoDataFrame({"name": named.name.values, "len": named.length_m.values,
                               "cx": (mid.x // cell_m).astype(int).values,
                               "cy": (mid.y // cell_m).astype(int).values},
                              geometry=mid.values, crs=UTM_EPSG)
        best = df.sort_values("len").groupby(["name", "cx", "cy"]).tail(1)
        # one street crossing a cell boundary is still one street: keep an
        # entry only if no longer entry of the same name lies within 3 km
        kept = {}
        for nm, g, ln in sorted(zip(best.name, best.geometry, best.len), key=lambda r: -r[2]):
            if all(g.distance(o) > 3000 for o in kept.get(nm, [])):
                kept.setdefault(nm, []).append(g)
        pts = gpd.GeoSeries([g for v in kept.values() for g in v], crs=UTM_EPSG).to_crs(4326)
        names = [nm for nm, v in kept.items() for _ in v]
        for nm, g in zip(names, pts):
            add(nm, g.x, g.y, "street")
    return {"kinds": KINDS, "items": items}


def web_bundle(edges, stations, lines, path, simplify_m=3.0, hours=None, gaz=None):
    """`hours` maps "HH:MM" to the network scored at that hour (from
    score.score_hours); its arrays let the map switch hour without a
    reload. Without it, only the base hour is exported."""
    nt = node_table(edges)
    idx = {n: i for i, n in enumerate(nt.index)}
    ll = edges.geometry.simplify(simplify_m).to_crs(4326)

    names, name_idx = [], {}
    shapes, rows = [], []
    for (_, e), g in zip(edges.iterrows(), ll.geometry):
        inner = list(g.coords)[1:-1]
        si = -1
        if inner:
            si = len(shapes)
            shapes.append([round(c, 5) for xy in inner for c in xy])
        nm = e["name"] or ""
        if nm not in name_idx:
            name_idx[nm] = len(names)
            names.append(nm)
        pct = lambda v: int(round(100 * float(v)))
        rows.append([idx[e.u], idx[e.v], round(float(e.length_m), 1),
                     pct(e.score), pct(e.confidence), pct(e.lighting),
                     pct(e.activity), pct(e.context), si, name_idx[nm],
                     int(e.lamps), round(float(e.open_expected), 1)])

    metro = None
    if stations is not None:
        sidx = {s: i for i, s in enumerate(stations.index)}
        metro = {
            "stations": [[round(r.lon, 5), round(r.lat, 5), r["name"]]
                         for _, r in stations.iterrows()],
            "lines": {k: [sidx[s] for s in v if s in sidx] for k, v in lines.items()},
            "colours": stations.attrs.get("colours", {}),
        }

    by_hour = []
    for label, h in (hours or {}).items():
        t = datetime.strptime(label, "%H:%M")
        by_hour.append({
            "t": label, "metro": metro_running(t),
            "score": [int(round(100 * v)) for v in h.score],
            "conf": [int(round(100 * v)) for v in h.confidence],
            "act": [int(round(100 * v)) for v in h.activity],
            "open": [round(float(v), 1) for v in h.open_expected],
            "mean_score": round(float(np.average(h.score, weights=h.length_m)), 3),
        })

    bundle = {
        "nodes": nt[["lon", "lat"]].round(5).values.tolist(),
        "edges": rows, "shapes": shapes, "names": names, "metro": metro,
        "hours": by_hour, "gaz": gaz,
        "params": {"walk_mps": WALK_SPEED_MPS, "beta": EXPOSURE_BETA,
                   "low_score": LOW_SCORE, "metro_mps": METRO_SPEED_MPS,
                   "dwell_s": METRO_DWELL_S, "wait_s": METRO_WAIT_S,
                   "transfer_s": TRANSFER_S, "access_m": STATION_ACCESS_M,
                   "metro_end": METRO_SERVICE_END, "metro_start": METRO_SERVICE_START,
                   "night_time": NIGHT_TIME.strftime("%H:%M"),
                   "weights": WEIGHTS},
        "stats": {"segments": len(edges),
                  "km": round(float(edges.length_m.sum()) / 1000, 1),
                  "mean_score": round(float(np.average(edges.score, weights=edges.length_m)), 3),
                  "mean_conf": round(float(np.average(edges.confidence, weights=edges.length_m)), 3)},
    }
    with open(path, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))
    return bundle
