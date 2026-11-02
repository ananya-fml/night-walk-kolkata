"""Night-walk score and confidence for each street segment.

score = w_L * lighting + w_A * activity + w_C * context (+ guardian bonus)

Each component is in [0, 1]. Each also carries a confidence in [0, 1]: how
much of it rests on direct observation (a lit=* tag, a parsed opening_hours)
rather than a prior (road class, category defaults, coarse satellite light).
The segment confidence is the weighted mean of the component confidences.

The score describes exposure: how lit, overlooked and busy a street is
likely to be at a given hour. It is not a crime prediction and should not
be presented as one.
"""
import numpy as np
import pandas as pd

from .config import (ACTIVITY_HALF, ACTIVITY_PRIOR, GUARDIAN_BONUS, ISOLATION_PENALTY,
                     LAMP_SPACING_FULL_M, LIT_PRIOR, ROAD_CONTEXT,
                     UNDERPASS_PENALTY, WEIGHTS)

LIT_YES = {"yes", "24/7", "automatic", "limited", "interval", "sunset-sunrise",
           "dusk-dawn"}


def lighting(e):
    tag = e.lit.str.lower().map(lambda v: 1.0 if v in LIT_YES
                                else 0.0 if v == "no" else np.nan)
    lamp = np.where(e.lamps > 0,
                    np.minimum(1.0, e.lamps * LAMP_SPACING_FULL_M / e.length_m.clip(lower=1)),
                    np.nan)
    prior = e.highway.map(LIT_PRIOR).fillna(0.4)
    # Satellite light only where the raster has data; elsewhere road class alone
    has_ntl = e.ntl.notna()
    ntl_pct = e.ntl.rank(pct=True)
    prior = prior.where(~has_ntl, 0.5 * prior + 0.5 * ntl_pct)
    observed = tag.where(tag.notna(), pd.Series(lamp, index=e.index))
    value = observed.where(observed.notna(), prior)
    conf = np.select([tag.notna(), ~np.isnan(lamp), has_ntl], [1.0, 0.7, 0.3],
                     default=0.15)
    return value.clip(0, 1), pd.Series(conf, index=e.index)


def activity(e, hour_factor=1.0):
    x = e.open_expected
    mapped = x / (x + ACTIVITY_HALF)
    # No mapped place nearby is not evidence of an empty street: OSM misses
    # most small shops. Use the road-class prior there, at low confidence.
    prior = (e.highway.map(ACTIVITY_PRIOR).fillna(0.05) * hour_factor).clip(upper=1)
    value = mapped.where(e.pois > 0, prior)
    share_obs = np.where(e.pois > 0, e.open_observed / e.pois.clip(lower=1), np.nan)
    conf = np.where(e.pois > 0, 0.2 + 0.8 * share_obs, 0.1)
    return value.clip(0, 1), pd.Series(conf, index=e.index)


def context(e):
    base = e.highway.map(ROAD_CONTEXT).fillna(0.4)
    v = base * (1 - ISOLATION_PENALTY * e.isolation)
    v = v - UNDERPASS_PENALTY * (e.tunnel | e.covered)
    return v.clip(0, 1), pd.Series(0.8, index=e.index)


def score(edges, weights=None, hour_factor=1.0):
    w = weights or WEIGHTS
    e = edges.copy()
    (e["lighting"], e["c_lighting"]) = lighting(e)
    (e["activity"], e["c_activity"]) = activity(e, hour_factor)
    (e["context"], e["c_context"]) = context(e)
    tot = sum(w.values())
    s = sum(w[k] * e[k] for k in ("lighting", "activity", "context")) / tot
    e["score"] = (s + GUARDIAN_BONUS * e.guardian).clip(0, 1)
    e["confidence"] = sum(w[k] * e[f"c_{k}"]
                          for k in ("lighting", "activity", "context")) / tot
    return e


def score_hours(edges, pois_raw, hours, weights=None):
    """Score the network at several hours. Lighting and context do not
    change with the hour; activity does."""
    from .features import hour_factor, open_at
    out = {}
    for when in hours:
        e = edges.copy()
        e["open_expected"], e["open_observed"] = open_at(edges, pois_raw, when)
        e["open_imputed"] = e.pois - e.open_observed
        out[when.strftime("%H:%M")] = score(e, weights, hour_factor(when))
    return out


def length_weighted(e, col):
    return float(np.average(e[col], weights=e.length_m))
