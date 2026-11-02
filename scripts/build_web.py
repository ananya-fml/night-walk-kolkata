"""Insert the scored network into the map page.

    python scripts/build_web.py web/data.json web/index.html \
        [--demo LON,LAT[,LABEL] LON,LAT[,LABEL]] [--synthetic] [--repo URL]
        [--tile-key CARTO_KEY]
"""
import argparse
import json
import os
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "web" / "template.html"


def build(data_path, out_path, demo=None, synthetic=False, repo=None, tile_key=None):
    data = json.loads(Path(data_path).read_text())
    data["synthetic"] = synthetic
    if demo:
        data["demo"] = demo
    if repo:
        data["repo"] = repo
    if tile_key:
        data["tileKey"] = tile_key   # CARTO basemap key; without it, Esri tiles
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*DATA*/null", blob)
    Path(out_path).write_text(html)
    return len(html)


def _pt(s):
    """"lon,lat" or "lon,lat,Label"."""
    parts = s.split(",", 2)
    pt = [float(parts[0]), float(parts[1])]
    return pt + [parts[2].strip()] if len(parts) == 3 else pt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--demo", nargs=2, type=_pt)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--repo")
    ap.add_argument("--tile-key", default=os.environ.get("CARTO_KEY"),
                    help="free CARTO basemap key (carto.com/basemaps/apikey); "
                         "omit to use Esri's keyless Dark Gray Canvas")
    a = ap.parse_args()
    n = build(a.data, a.out, a.demo, a.synthetic, a.repo, a.tile_key)
    print(f"wrote {a.out} ({n/1e6:.2f} MB)")
