"""Metro stations and line sequences from OSM route relations.

Stations are railway=station nodes tagged as metro (station=subway,
subway=yes or light_rail). Each route=subway relation gives an ordered list
of stop members; stops are snapped to the nearest station within 300 m. The
two directions of a line share a ref, so the longer one is kept.
"""
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from .config import UTM_EPSG

_TO_UTM = Transformer.from_crs(4326, UTM_EPSG, always_xy=True)
STOP_ROLES = {"stop", "stop_entry_only", "stop_exit_only", "station", "platform"}


def _is_metro_station(t):
    return (t.get("railway") == "station"
            and (t.get("station") in {"subway", "light_rail"}
                 or t.get("subway") == "yes" or t.get("light_rail") == "yes"))


def parse(transit, snap_m=300):
    el = transit["elements"]
    nodes = {e["id"]: e for e in el if e["type"] == "node" and "lat" in e}
    st = [e for e in nodes.values() if _is_metro_station(e.get("tags", {}))]
    if not st:
        return None, {}
    stations = pd.DataFrame({
        "name": [e["tags"].get("name:en") or e["tags"].get("name", str(e["id"])) for e in st],
        "lon": [e["lon"] for e in st], "lat": [e["lat"] for e in st],
    }, index=[e["id"] for e in st])
    stations["x"], stations["y"] = _TO_UTM.transform(stations.lon.values, stations.lat.values)
    tree = cKDTree(stations[["x", "y"]].values)

    lines = {}
    colours = {}
    for r in el:
        t = r.get("tags", {})
        if r["type"] != "relation" or t.get("route") not in {"subway", "light_rail"}:
            continue
        key = t.get("ref") or t.get("name") or str(r["id"])
        seq = []
        for m in r.get("members", []):
            if m["type"] != "node" or m.get("role", "") not in STOP_ROLES:
                continue
            n = nodes.get(m["ref"])
            if n is None:
                continue
            x, y = _TO_UTM.transform(n["lon"], n["lat"])
            d, k = tree.query([x, y])
            if d <= snap_m:
                sid = stations.index[k]
                if not seq or seq[-1] != sid:
                    seq.append(sid)
        if len(seq) >= 2 and len(seq) > len(lines.get(key, [])):
            lines[key] = seq
            colours[key] = t.get("colour", "")

    # Directions of one line can carry different refs ("Blue" vs "Blue Line
    # (A -> B)"). Drop any line whose stations are all served by a longer one.
    for k in sorted(lines, key=lambda k: len(lines[k])):
        if any(o != k and set(lines[k]) <= set(lines[o]) for o in lines):
            lines.pop(k)
            colours.pop(k, None)
    stations["lines"] = [[k for k, s in lines.items() if sid in s]
                         for sid in stations.index]
    stations = stations[stations.lines.map(len) > 0]
    stations.attrs["colours"] = colours
    return stations, lines


def summary(stations, lines):
    if stations is None:
        return "no metro stations found"
    return {k: f"{len(v)} stations: {stations.at[v[0], 'name']} - "
               f"{stations.at[v[-1], 'name']}" for k, v in lines.items()}
