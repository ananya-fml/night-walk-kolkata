"""Raw OSM snapshot -> scored street segments -> web map data.

    python -m nightwalk.pipeline [--ntl path/to/kolkata_ntl_median_500m.tif]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import features, fetch, network, score, transit
from .config import BBOX, NIGHT_TIME, WEIGHTS

ROOT = Path(__file__).resolve().parents[2]


def run(raw=fetch.RAW, ntl=None, out=ROOT / "data" / "processed", when=NIGHT_TIME,
        verbose=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    edges = network.build_edges(fetch.load("streets", raw))
    s = network.summary(edges)
    say(f"network   {s['segments']:,} segments, {s['km']:,} km, "
        f"lit=* on {s['lit_tagged_pct']}%")

    pois = features.pois_frame(fetch.load("pois", raw), when)
    lamps = features.lamps_frame(fetch.load("lamps", raw))
    land = features.landuse_frame(fetch.load("landuse", raw))
    say(f"inputs    {len(pois):,} places, {len(lamps):,} lamps, "
        f"{len(land):,} isolating land parcels")

    edges = features.add_features(edges, pois, lamps, land, ntl)
    edges = score.score(edges)
    cov = features.coverage(edges, pois)
    say("coverage\n" + cov.to_string())
    say(f"score     length-weighted mean {score.length_weighted(edges, 'score'):.3f}, "
        f"confidence {score.length_weighted(edges, 'confidence'):.3f}")

    stations, lines = transit.parse(fetch.load("transit", raw))
    say(f"metro     {transit.summary(stations, lines)}")

    edges.to_file(out / "segments.gpkg", driver="GPKG")
    meta = {"night_time": when.isoformat(), "bbox": list(BBOX), "weights": WEIGHTS,
            "coverage": cov.to_dict(),
            "osm_snapshot": fetch.load("streets", raw).get("_fetched")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return edges, pois, stations, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntl", help="VIIRS radiance GeoTIFF covering the study area")
    ap.add_argument("--raw", default=str(fetch.RAW))
    ap.add_argument("--web", default=str(ROOT / "web" / "data.json"))
    a = ap.parse_args()
    edges, pois, stations, lines = run(a.raw, a.ntl)
    from .config import NIGHT_HOURS
    from .export import web_bundle
    hours = score.score_hours(edges, fetch.load("pois", a.raw), NIGHT_HOURS)
    for t, h in hours.items():
        print(f"{t}  mean score {score.length_weighted(h, 'score'):.3f}")
    from .export import gazetteer
    places = fetch.load("places", a.raw) if (Path(a.raw) / "places.json").exists() else None
    gaz = gazetteer(edges, pois, stations, fetch.load("transit", a.raw), places)
    web_bundle(edges, stations, lines, a.web, hours=hours, gaz=gaz)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
