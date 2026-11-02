"""How much should anyone trust the score? Checks that need no ground truth.

1. Route-level effect: over random origin-destination pairs, how often the
   safer route differs from the shortest, the detour it costs and the exposed
   distance it saves.
2. Weight sensitivity: perturb the component weights and measure how stable
   the segment ranking and the chosen routes are.
3. Convergent validity: where both exist, does the OSM lit=* tag agree with
   VIIRS radiance (run only with 10+ streets in each group)? Do the
   components carry independent information?
4. Coverage bias: is confidence lower in some parts of the city than others?
   A score that is mostly guessed in poorer or peripheral areas would
   systematically misdirect people there.

    python -m nightwalk.evaluate          # after the pipeline has run
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from .config import EXPOSURE_BETA, WEIGHTS
from .route import Router
from .score import score as score_fn


def od_pairs(edges, n=200, min_m=800, max_m=4000, seed=7):
    rng = np.random.default_rng(seed)
    R = Router(edges)
    xy = R.nodes[["x", "y"]].values
    ids = R.ids
    out = []
    while len(out) < n:
        a, b = rng.integers(0, len(ids), 2)
        d = np.hypot(*(xy[a] - xy[b]))
        if min_m <= d <= max_m:
            out.append((ids[a], ids[b]))
    return out


def route_effect(edges, pairs, beta=EXPOSURE_BETA):
    R = Router(edges)
    rows = []
    for o, d in pairs:
        s = R.route(o, d, beta=0.0)
        f = R.route(o, d, beta=beta)
        rows.append({"shortest_m": s.walk_m, "safer_m": f.walk_m,
                     "detour_pct": 100 * (f.walk_m / s.walk_m - 1),
                     "exposed_short_m": s.exposed_m, "exposed_safe_m": f.exposed_m,
                     "score_short": s.mean_score, "score_safe": f.mean_score,
                     "differs": set(s.edges) != set(f.edges),
                     "conf_short": s.mean_confidence})
    return pd.DataFrame(rows)


def weight_sensitivity(edges, pairs, step=0.15, beta=EXPOSURE_BETA):
    """Rank stability and route overlap when each weight moves by +/- step."""
    base_routes = route_paths(edges, pairs, beta)
    rows = []
    for k, sign in itertools.product(WEIGHTS, (-1, 1)):
        w = dict(WEIGHTS)
        w[k] = max(0.0, w[k] + sign * step)
        alt = score_fn(edges, w)
        rho = spearmanr(edges.score, alt.score).statistic
        alt_routes = route_paths(alt, pairs, beta)
        jac = [len(a & b) / max(len(a | b), 1) for a, b in zip(base_routes, alt_routes)]
        rows.append({"weight": k, "change": f"{sign*step:+.2f}",
                     "segment_rank_rho": round(rho, 3),
                     "route_overlap_mean": round(float(np.mean(jac)), 3),
                     "routes_unchanged_pct": round(100 * np.mean(np.isclose(jac, 1)), 1)})
    return pd.DataFrame(rows)


def route_paths(edges, pairs, beta):
    R = Router(edges)
    return [set(R.route(o, d, beta=beta).edges) for o, d in pairs]


def convergent(edges):
    e = edges
    out = {}
    tagged = e[e.lit.isin(["yes", "no"]) & e.ntl.notna()]
    ways = tagged.groupby("lit").way_id.nunique()
    out["streets tagged lit=yes / lit=no"] = f"{ways.get('yes', 0)} / {ways.get('no', 0)}"
    # Segments of one street are not independent; require enough streets.
    if ways.get("yes", 0) >= 10 and ways.get("no", 0) >= 10:
        yes, no = tagged[tagged.lit == "yes"].ntl, tagged[tagged.lit == "no"].ntl
        u = mannwhitneyu(yes, no, alternative="greater")
        out["ntl median, lit=yes"] = round(float(yes.median()), 2)
        out["ntl median, lit=no"] = round(float(no.median()), 2)
        out["AUC lit=yes > lit=no"] = round(u.statistic / (len(yes) * len(no)), 3)
        out["n tagged"] = len(tagged)
    else:
        out["lit vs night-light test"] = "not run: fewer than 10 streets in a group"
    comp = e[["lighting", "activity", "context"]].corr(method="spearman").round(3)
    out["component correlations"] = comp
    return out


def coverage_by_area(edges, cell_m=1000):
    """Mean confidence on a coarse grid; the spread is the coverage bias."""
    mid = edges.geometry.interpolate(0.5, normalized=True)
    g = pd.DataFrame({"cx": (mid.x // cell_m).astype(int), "cy": (mid.y // cell_m).astype(int),
                      "conf": edges.confidence, "len": edges.length_m,
                      "ntl": edges.ntl})
    a = g.groupby(["cx", "cy"]).apply(
        lambda d: pd.Series({"confidence": np.average(d.conf, weights=d.len),
                             "km": d.len.sum() / 1000, "ntl": d.ntl.median()}),
        include_groups=False)
    a = a[a.km >= 1]
    res = {"cells": len(a),
           "confidence p10": round(a.confidence.quantile(.1), 3),
           "confidence p90": round(a.confidence.quantile(.9), 3)}
    if a.ntl.notna().sum() > 10:
        res["confidence vs night light (rho)"] = round(
            spearmanr(a.confidence, a.ntl, nan_policy="omit").statistic, 3)
    return res, a


def run(edges, n_pairs=200, verbose=True):
    pairs = od_pairs(edges, n_pairs)
    eff = route_effect(edges, pairs)
    sens = weight_sensitivity(edges, pairs[: min(60, len(pairs))])
    conv = convergent(edges)
    cov, cells = coverage_by_area(edges)
    if verbose:
        d = eff
        print(f"\nROUTE EFFECT ({len(d)} random trips, 0.8-4 km)")
        print(f"  safer route differs from shortest: {100*d.differs.mean():.0f}%")
        print(f"  median detour: {d.detour_pct.median():.1f}%  (p90 {d.detour_pct.quantile(.9):.1f}%)")
        print(f"  exposed distance: median {d.exposed_short_m.median():.0f} m -> "
              f"{d.exposed_safe_m.median():.0f} m")
        print("\nWEIGHT SENSITIVITY (each weight +/-0.15)")
        print(sens.to_string(index=False))
        print("\nCONVERGENT CHECKS")
        for k, v in conv.items():
            print(f"  {k}:\n{v}" if isinstance(v, pd.DataFrame) else f"  {k}: {v}")
        print("\nCOVERAGE BY AREA (1 km cells)")
        for k, v in cov.items():
            print(f"  {k}: {v}")
    return {"effect": eff, "sensitivity": sens, "convergent": conv,
            "coverage": cov, "cells": cells}


if __name__ == "__main__":
    import geopandas as gpd
    from .pipeline import ROOT
    edges = gpd.read_file(ROOT / "data" / "processed" / "segments.gpkg")
    for c in ("tunnel", "covered", "bridge", "guardian"):
        edges[c] = edges[c].astype(bool)
    edges["lit"] = edges.lit.fillna("")
    edges["name"] = edges.name.fillna("")
    r = run(edges)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    r["effect"].to_csv(out / "route_effect.csv", index=False)
    r["sensitivity"].to_csv(out / "weight_sensitivity.csv", index=False)
    r["cells"].to_csv(out / "confidence_by_cell.csv")
