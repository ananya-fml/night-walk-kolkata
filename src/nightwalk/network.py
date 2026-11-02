"""Walkable street network from an Overpass response.

Ways are split at every node shared by two or more walkable ways, giving one
edge per street segment between junctions. The graph is undirected (people
walk both ways down one-way streets) and only the largest connected component
is kept.
"""
from collections import Counter

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString

from .config import EXCLUDE_FOOT, UTM_EPSG, WALKABLE


def _walkable(tags):
    hw = tags.get("highway")
    if hw not in WALKABLE:
        return False
    if tags.get("foot") in EXCLUDE_FOOT:
        return False
    if tags.get("access") in {"private", "no"} and tags.get("foot") not in {"yes", "designated"}:
        return False
    if tags.get("area") == "yes":
        return False
    return True


def build_edges(streets):
    """Return a GeoDataFrame of street segments in UTM, one row per edge."""
    nodes = {e["id"]: (e["lon"], e["lat"])
             for e in streets["elements"] if e["type"] == "node"}
    ways = [e for e in streets["elements"]
            if e["type"] == "way" and _walkable(e.get("tags", {}))]

    use = Counter()
    for w in ways:
        nd = w["nodes"]
        use.update(nd)
        use[nd[0]] += 1          # endpoints are always split points
        use[nd[-1]] += 1

    rows = []
    for w in ways:
        t = w.get("tags", {})
        nd = [n for n in w["nodes"] if n in nodes]
        if len(nd) < 2:
            continue
        start = 0
        for i in range(1, len(nd)):
            if use[nd[i]] > 1 or i == len(nd) - 1:
                seg = nd[start:i + 1]
                if len(seg) >= 2 and seg[0] != seg[-1]:
                    rows.append({
                        "u": seg[0], "v": seg[-1], "way_id": w["id"],
                        "highway": t.get("highway"),
                        "name": t.get("name", ""),
                        "lit": t.get("lit", ""),
                        "tunnel": t.get("tunnel", "") not in {"", "no"},
                        "covered": t.get("covered", "") not in {"", "no"},
                        "bridge": t.get("bridge", "") not in {"", "no"},
                        "sidewalk": t.get("sidewalk", ""),
                        "geometry": LineString([nodes[n] for n in seg]),
                    })
                start = i
    g = gpd.GeoDataFrame(rows, crs=4326).to_crs(UTM_EPSG)
    g["length_m"] = g.length
    g = g[g.length_m > 0.5]

    # largest connected component
    G = nx.Graph()
    G.add_edges_from(zip(g.u, g.v))
    giant = max(nx.connected_components(G), key=len)
    keep = g.u.isin(giant) & g.v.isin(giant)
    g = g[keep].reset_index(drop=True)
    g["edge_id"] = range(len(g))
    return g


def node_table(edges):
    """Junction coordinates (lon, lat and UTM x, y) for every graph node."""
    ll = edges.to_crs(4326)
    recs = {}
    for (u, v), gl, gu in zip(zip(edges.u, edges.v), ll.geometry, edges.geometry):
        cl, cu = list(gl.coords), list(gu.coords)
        recs.setdefault(u, (*cl[0], *cu[0]))
        recs.setdefault(v, (*cl[-1], *cu[-1]))
    return pd.DataFrame.from_dict(recs, orient="index",
                                  columns=["lon", "lat", "x", "y"])


def summary(edges):
    return {
        "segments": len(edges),
        "km": round(edges.length_m.sum() / 1000, 1),
        "junctions": len(set(edges.u) | set(edges.v)),
        "lit_tagged_pct": round(100 * (edges.lit != "").mean(), 1),
        "by_class": edges.groupby("highway").length_m.sum()
                         .div(1000).round(1).sort_values(ascending=False)
                         .to_dict(),
    }
