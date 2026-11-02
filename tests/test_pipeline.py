from datetime import datetime

import networkx as nx
import numpy as np
import pytest

from nightwalk import features, pipeline, route, score
from nightwalk.config import NIGHT_TIME

import synthetic


@pytest.fixture(scope="module")
def city(tmp_path_factory):
    raw = synthetic.write(tmp_path_factory.mktemp("raw"))
    out = tmp_path_factory.mktemp("out")
    edges, pois, stations, lines = pipeline.run(raw, out=out, verbose=False)
    return {"raw": raw, "edges": edges, "pois": pois,
            "stations": stations, "lines": lines}


# ── network ─────────────────────────────────────────────────────────────
def test_grid_is_split_at_every_junction(city):
    e = city["edges"]
    grid = 2 * 12 * 11                      # 12 rows x 11 links, both directions
    path = 5                                # diagonal footway, 6 nodes
    assert len(e) == grid + path            # private service road excluded
    assert e.length_m.between(190, 290).all()


def test_private_road_is_excluded(city):
    assert not (city["edges"].highway == "service").any()


# ── opening hours ───────────────────────────────────────────────────────
@pytest.mark.parametrize("tags,expected", [
    ({"amenity": "restaurant", "opening_hours": "Mo-Su 11:00-23:30"}, (1.0, True)),
    ({"shop": "clothes", "opening_hours": "Mo-Sa 10:00-20:00"}, (0.0, True)),
    ({"amenity": "police", "opening_hours": "24/7"}, (1.0, True)),
    ({"amenity": "bar", "opening_hours": "18:00-02:00"}, (1.0, True)),
    ({"amenity": "bar", "opening_hours": "nonsense"}, (0.8, False)),
    ({"shop": "convenience"}, (0.5, False)),
    ({"shop": "hardware"}, (0.25, False)),
])
def test_open_status(tags, expected):
    assert features.open_status(tags, NIGHT_TIME) == expected


def test_closed_shop_counts_at_eight_pm_not_ten():
    tags = {"shop": "clothes", "opening_hours": "Mo-Sa 10:00-20:00"}
    assert features.open_status(tags, datetime(2026, 11, 4, 19, 0))[0] == 1.0
    assert features.open_status(tags, NIGHT_TIME)[0] == 0.0


def test_furniture_is_not_a_place(city):
    assert "bench" not in set(city["pois"].category)


# ── features and score ──────────────────────────────────────────────────
def test_main_road_outscores_dark_lane_and_park_path(city):
    e = city["edges"]
    main = e[e.name == "Main Road"].score.mean()
    lane = e[e.name == "Dark Lane"].score.mean()
    path = e[e.highway == "footway"].score.mean()
    assert main > lane > path


def test_park_marks_the_diagonal_as_isolated(city):
    e = city["edges"]
    assert e[e.highway == "footway"].isolation.min() > 0.8
    assert e[e.name == "Main Road"].isolation.max() == 0


def test_lit_tag_gives_full_confidence_and_prior_gives_less(city):
    e = city["edges"]
    assert (e[e.lit != ""].c_lighting == 1.0).all()
    assert (e[(e.lit == "") & (e.lamps == 0)].c_lighting < 0.5).all()


def test_scores_and_confidence_in_unit_interval(city):
    e = city["edges"]
    for c in ("score", "confidence", "lighting", "activity", "context"):
        assert e[c].between(0, 1).all(), c


def test_weights_change_scores(city):
    e = city["edges"]
    lit_heavy = score.score(e, {"lighting": 1, "activity": 0, "context": 0})
    assert not np.allclose(lit_heavy.score, e.score)


# ── routing ─────────────────────────────────────────────────────────────
def node(i, j):
    return synthetic.nid(i, j)


def test_beta_zero_is_the_length_shortest_path(city):
    e = city["edges"]
    r = route.Router(e).route(node(1, 1), node(6, 6), beta=0.0)
    G = nx.Graph()
    G.add_weighted_edges_from(zip(e.u, e.v, e.length_m))
    best = nx.shortest_path_length(G, node(1, 1), node(6, 6), weight="weight")
    assert r.walk_m == pytest.approx(best, rel=1e-6)


def test_shortest_takes_the_dark_park_path_and_safer_avoids_it(city):
    e = city["edges"]
    R = route.Router(e)
    short, safe = R.compare(node(1, 1), node(6, 6), with_metro=False)
    used = lambda r: set(e.set_index("edge_id").loc[r.edges].highway)
    assert "footway" in used(short)
    assert "footway" not in used(safe)
    assert safe.walk_m > short.walk_m
    assert safe.mean_score > short.mean_score


def test_metro_parses_one_line_with_three_stations(city):
    assert list(city["lines"]) == ["Purple"]
    assert len(city["lines"]["Purple"]) == 3


def test_metro_used_before_close_and_not_after(city):
    o, d = node(11, 0), node(11, 11)
    before = route.Router(city["edges"], city["stations"], city["lines"],
                          when=datetime(2026, 11, 4, 22, 0))
    r = before.route(o, d, metro=True)
    assert any(leg[0] == "metro" for leg in r.legs) and r.ride_s > 0
    for late in (datetime(2026, 11, 4, 23, 0), datetime(2026, 11, 5, 0, 30)):
        after = route.Router(city["edges"], city["stations"], city["lines"], when=late)
        r = after.route(o, d, metro=True)
        assert not any(leg[0] == "metro" for leg in r.legs), late


def test_activity_falls_after_midnight_but_lighting_does_not(city, tmp_path):
    from nightwalk import fetch
    from nightwalk.config import NIGHT_HOURS
    hours = score.score_hours(city["edges"], fetch.load("pois", city["raw"]), NIGHT_HOURS)
    act = [score.length_weighted(h, "activity") for h in hours.values()]
    assert act == sorted(act, reverse=True) and act[0] > act[-1]
    lit = [score.length_weighted(h, "lighting") for h in hours.values()]
    assert np.allclose(lit, lit[0])


def test_parsed_hours_beat_priors_at_any_hour():
    tags = {"amenity": "restaurant", "opening_hours": "Mo-Su 11:00-23:30"}
    assert features.open_status(tags, datetime(2026, 11, 5, 0, 30)) == (0.0, True)
    assert features.open_status({"amenity": "hospital"}, datetime(2026, 11, 5, 2, 0))[0] == 1.0
    assert features.open_status({"amenity": "restaurant"}, datetime(2026, 11, 5, 2, 0))[0] < 0.2


def test_web_bundle_round_trip(city, tmp_path):
    from nightwalk import fetch
    from nightwalk.config import NIGHT_HOURS
    from nightwalk.export import web_bundle
    hours = score.score_hours(city["edges"], fetch.load("pois", city["raw"]), NIGHT_HOURS)
    b = web_bundle(city["edges"], city["stations"], city["lines"], tmp_path / "d.json",
                   hours=hours)
    assert [h["t"] for h in b["hours"]] == ["20:00", "22:00", "00:00", "02:00"]
    assert [h["metro"] for h in b["hours"]] == [True, True, False, False]
    assert all(len(h["score"]) == len(city["edges"]) for h in b["hours"])
    assert len(b["edges"]) == len(city["edges"])
    u, v = b["edges"][0][:2]
    assert 0 <= u < len(b["nodes"]) and 0 <= v < len(b["nodes"])
    assert b["metro"]["lines"]["Purple"] == [0, 1, 2] or len(b["metro"]["lines"]["Purple"]) == 3


def test_unmapped_street_gets_activity_prior_not_zero(city):
    e = city["edges"]
    bare = e[(e.pois == 0) & (e.highway == "residential")]
    assert len(bare) and (bare.activity > 0).all()
    assert (bare.c_activity < 0.2).all()


def test_duplicate_line_directions_are_merged():
    from nightwalk import transit as tr
    raw = synthetic.build()["transit"]
    rel = next(x for x in raw["elements"] if x["type"] == "relation")
    dup = dict(rel, id=90_002, tags={"route": "subway", "name": "Purple Line (C -> A)"},
               members=list(reversed(rel["members"])))
    raw["elements"].append(dup)
    stations, lines = tr.parse(raw)
    assert len(lines) == 1


def test_zero_radiance_is_treated_as_missing(tmp_path, city):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    e = city["edges"].to_crs(4326)
    w, s, east, n = e.total_bounds
    path = tmp_path / "ntl.tif"
    arr = np.zeros((4, 4), "float32")
    arr[:2] = 40.0                          # north half lit, south half zero-filled
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(w, s, east, n, 4, 4)) as dst:
        dst.write(arr, 1)
    vals = features.sample_raster(city["edges"], path)
    assert np.isnan(vals).any() and (vals[~np.isnan(vals)] == 40).all()


def test_gazetteer_has_streets_places_and_stations(city):
    from nightwalk import fetch
    from nightwalk.export import gazetteer
    g = gazetteer(city["edges"], city["pois"], city["stations"],
                  fetch.load("transit", city["raw"]))
    names = {i[0] for i in g["items"]}
    assert {"Main Road", "Lit Street", "Station A metro"} <= names
    kinds = {g["kinds"][i[3]] for i in g["items"]}
    assert {"street", "metro"} <= kinds
    # one entry per street name here: the grid is smaller than 3 km
    assert sum(i[0] == "Main Road" for i in g["items"]) == 1
