"""A small synthetic city in Overpass JSON format, for tests.

12 x 12 street grid, 200 m blocks, centred on central Kolkata:

  row 0     primary road, lit=yes, lamps, restaurants open until 23:30
  row 1     residential, lit=yes ("Lit Street")
  col 6     residential, lit=yes ("Lit Avenue")
  col 0     residential, lit=no ("Dark Lane")
  park      a narrow strip around an unlit footway from (1,1) to (6,6):
            1.4 km through the park versus a 2.0 km lit detour along
            row 1 and col 6
  others    residential, lit unknown
  metro     three stations along row 11 (north), one line
"""
import json
import math
from pathlib import Path

LAT0, LON0 = 22.55, 88.35
STEP = 200.0
N = 12


def ll(i, j):
    """Grid (row i north, col j east) -> lon, lat."""
    lat = LAT0 + i * STEP / 111_320
    lon = LON0 + j * STEP / (111_320 * math.cos(math.radians(LAT0)))
    return round(lon, 7), round(lat, 7)


def nid(i, j):
    return 1_000 + i * N + j


def build():
    nodes, ways, wid = {}, [], 10_000
    for i in range(N):
        for j in range(N):
            lon, lat = ll(i, j)
            nodes[nid(i, j)] = {"type": "node", "id": nid(i, j), "lat": lat, "lon": lon}

    def way(nds, **tags):
        nonlocal wid
        wid += 1
        ways.append({"type": "way", "id": wid, "nodes": nds, "tags": tags})

    for i in range(N):                                   # east-west streets
        tags = ({"highway": "primary", "lit": "yes", "name": "Main Road"} if i == 0
                else {"highway": "residential", "lit": "yes", "name": "Lit Street"} if i == 1
                else {"highway": "residential"})
        way([nid(i, j) for j in range(N)], **tags)
    for j in range(N):                                   # north-south streets
        tags = ({"highway": "residential", "lit": "no", "name": "Dark Lane"} if j == 0
                else {"highway": "residential", "lit": "yes", "name": "Lit Avenue"} if j == 6
                else {"highway": "residential"})
        way([nid(i, j) for i in range(N)], **tags)
    # unlit diagonal shortcut through the park, (1,1) -> (6,6)
    way([nid(k, k) for k in range(1, 7)], highway="footway", lit="no")
    # a private road that must be excluded
    way([nid(3, 8), nid(4, 9)], highway="service", access="private")

    streets = {"elements": list(nodes.values()) + ways}

    lamps = {"elements": [{"type": "node", "id": 50_000 + j,
                           "lat": ll(0, j)[1] + 0.00005, "lon": ll(0, j)[0] + 0.0006}
                          for j in range(N - 1)]}

    pois_el, pid = [], 60_000
    for j in range(0, N - 1, 2):                         # along the main road
        lon, lat = ll(0, j)
        pid += 1
        pois_el.append({"type": "node", "id": pid, "lat": lat + 0.0002, "lon": lon + 0.001,
                        "tags": {"amenity": "restaurant",
                                 "opening_hours": "Mo-Su 11:00-23:30"}})
        pid += 1
        pois_el.append({"type": "node", "id": pid, "lat": lat + 0.0002, "lon": lon + 0.0012,
                        "tags": {"shop": "clothes", "opening_hours": "Mo-Sa 10:00-20:00"}})
    lon, lat = ll(8, 8)
    pois_el.append({"type": "node", "id": 69_001, "lat": lat, "lon": lon + 0.0003,
                    "tags": {"shop": "convenience"}})                       # no hours
    pois_el.append({"type": "node", "id": 69_002, "lat": lat, "lon": lon,
                    "tags": {"amenity": "bench"}})                          # furniture
    lon, lat = ll(5, 10)
    pois_el.append({"type": "node", "id": 69_003, "lat": lat, "lon": lon + 0.0004,
                    "tags": {"amenity": "police", "opening_hours": "24/7"}})
    pois = {"elements": pois_el}

    # a narrow park strip around the diagonal footway (~35 m either side)
    from shapely.geometry import LineString
    strip = LineString([ll(k, k) for k in range(1, 7)]).buffer(0.0003, cap_style=2)
    landuse = {"elements": [{
        "type": "way", "id": 70_001, "tags": {"leisure": "park"},
        "geometry": [{"lon": x, "lat": y} for x, y in strip.exterior.coords]}]}

    st_el, members = [], []
    for k, j in enumerate((0, 6, 11)):
        lon, lat = ll(11, j)
        sid = 80_000 + k
        st_el.append({"type": "node", "id": sid, "lat": lat + 0.0003, "lon": lon,
                      "tags": {"railway": "station", "station": "subway",
                               "name": f"Station {'ABC'[k]}"}})
        st_el.append({"type": "node", "id": sid + 100, "lat": lat + 0.0004, "lon": lon})
        members.append({"type": "node", "ref": sid + 100, "role": "stop"})
    st_el.append({"type": "relation", "id": 90_001, "members": members,
                  "tags": {"route": "subway", "ref": "Purple", "colour": "#7b2d8e"}})
    transit = {"elements": st_el}

    return {"streets": streets, "lamps": lamps, "pois": pois,
            "landuse": landuse, "transit": transit}


def write(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in build().items():
        (folder / f"{k}.json").write_text(json.dumps(v))
    return folder
