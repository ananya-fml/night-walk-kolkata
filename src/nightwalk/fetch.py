"""Download OpenStreetMap layers for the study area through the Overpass API.

    python -m nightwalk.fetch            # all layers into data/raw/

Raw responses are cached as JSON, so the rest of the pipeline is offline and
reproducible from a dated snapshot (the timestamp is stored with each file).
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import BBOX, ISOLATING_LANDUSE

MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter"]
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _bbox():
    s, w, n, e = BBOX
    return f"{s},{w},{n},{e}"


def queries():
    b = _bbox()
    landuse = "\n".join(
        f'  way["{k}"="{v}"]({b});\n  relation["{k}"="{v}"]({b});'
        for k, v in sorted(ISOLATING_LANDUSE))
    return {
        "streets": f"""
[out:json][timeout:300];
way["highway"]({b});
out body;
>;
out skel qt;""",
        "lamps": f"""
[out:json][timeout:120];
node["highway"="street_lamp"]({b});
out;""",
        "pois": f"""
[out:json][timeout:300];
(
  nwr["amenity"]({b});
  nwr["shop"]({b});
  nwr["tourism"~"hotel|guest_house|hostel"]({b});
  nwr["leisure"~"sports_centre|fitness_centre"]({b});
);
out center tags;""",
        "transit": f"""
[out:json][timeout:180];
(
  relation["route"~"subway|light_rail"]({b});
  node["railway"="station"]({b});
  node["public_transport"="stop_position"]["subway"="yes"]({b});
  node["railway"="subway_entrance"]({b});
  node["highway"="bus_stop"]({b});
);
out body;
>;
out skel qt;""",
        "places": f"""
[out:json][timeout:180];
(
  node["place"~"suburb|neighbourhood|quarter|locality"]({b});
  nwr["tourism"~"attraction|museum|viewpoint|gallery"]({b});
  nwr["historic"]["name"]({b});
  nwr["leisure"="park"]["name"]({b});
);
out center tags;""",
        "landuse": f"""
[out:json][timeout:300];
(
{landuse}
);
out geom;""",
    }


def overpass(query, retries=3):
    last = None
    for attempt in range(retries):
        for url in MIRRORS:
            try:
                r = requests.post(url, data={"data": query}, timeout=400,
                                  headers={"User-Agent": "night-walk-kolkata/0.1"})
                if r.status_code == 200:
                    return r.json()
                last = f"{url}: HTTP {r.status_code}"
            except requests.RequestException as e:
                last = f"{url}: {e}"
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {last}")


def fetch_all(layers=None, force=False):
    RAW.mkdir(parents=True, exist_ok=True)
    for name, q in queries().items():
        if layers and name not in layers:
            continue
        out = RAW / f"{name}.json"
        if out.exists() and not force:
            print(f"{name:9s} cached")
            continue
        t0 = time.time()
        data = overpass(q)
        data["_fetched"] = datetime.now(timezone.utc).isoformat()
        data["_bbox"] = list(BBOX)
        out.write_text(json.dumps(data))
        print(f"{name:9s} {len(data['elements']):>8,} elements "
              f"({out.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f} s)")


def load(name, raw=RAW):
    return json.loads((Path(raw) / f"{name}.json").read_text())


if __name__ == "__main__":
    import sys
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    fetch_all(names or None, force="--force" in sys.argv)
