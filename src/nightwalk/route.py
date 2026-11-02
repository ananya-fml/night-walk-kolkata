"""Shortest vs safer walking routes, and walk + metro.

Walking cost is generalised time:

    walk_seconds * (1 + beta * (1 - score))

so beta = 0 gives the shortest path and larger beta trades distance for
better-lit, busier streets. Metro legs cost wait + in-vehicle time and are
available only before METRO_SERVICE_END.
"""
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

from .config import (EXPOSURE_BETA, LOW_SCORE, METRO_DWELL_S,
                     METRO_SERVICE_END, METRO_SERVICE_START, METRO_SPEED_MPS, METRO_WAIT_S,
                     NIGHT_TIME, STATION_ACCESS_M, TRANSFER_S, UTM_EPSG,
                     WALK_SPEED_MPS)
from .network import node_table

_TO_UTM = Transformer.from_crs(4326, UTM_EPSG, always_xy=True)


@dataclass
class Route:
    kind: str
    nodes: list
    edges: list = field(default_factory=list)      # edge_ids walked
    legs: list = field(default_factory=list)       # (mode, description)
    walk_m: float = 0.0
    ride_s: float = 0.0
    wait_s: float = 0.0
    mean_score: float = np.nan
    min_score: float = np.nan
    exposed_m: float = 0.0
    mean_confidence: float = np.nan

    @property
    def minutes(self):
        return (self.walk_m / WALK_SPEED_MPS + self.ride_s + self.wait_s) / 60

    def summary(self):
        return {"route": self.kind, "walk_km": round(self.walk_m / 1000, 2),
                "minutes": round(self.minutes, 1),
                "mean_score": round(self.mean_score, 3),
                "min_score": round(self.min_score, 3),
                "exposed_m": round(self.exposed_m),
                "exposed_share": round(self.exposed_m / max(self.walk_m, 1), 3),
                "confidence": round(self.mean_confidence, 2)}


def _hm(s):
    h, m = map(int, s.split(":"))
    return h * 60 + m


def metro_running(when, start=None, end=None):
    """True if `when` falls inside the daily service window."""
    t = when.hour * 60 + when.minute
    return _hm(start or METRO_SERVICE_START) <= t < _hm(end or METRO_SERVICE_END)


class Router:
    def __init__(self, edges, stations=None, lines=None, when=NIGHT_TIME):
        self.when = when
        self.e = edges.set_index("edge_id", drop=False)
        self.nodes = node_table(edges)
        self.tree = cKDTree(self.nodes[["x", "y"]].values)
        self.ids = self.nodes.index.values
        self.stations = stations
        self.lines = lines or {}
        self._cache = {}

    # ── graph construction ─────────────────────────────────────────────
    def _walk_graph(self, beta):
        G = nx.DiGraph()
        t = self.e.length_m / WALK_SPEED_MPS
        cost = t * (1 + beta * (1 - self.e.score))
        for eid, u, v, c in zip(self.e.edge_id, self.e.u, self.e.v, cost):
            for a, b in ((u, v), (v, u)):
                if not G.has_edge(a, b) or c < G[a][b]["cost"]:
                    G.add_edge(a, b, cost=c, eid=eid, mode="walk")
        return G

    def metro_available(self, when=None):
        return metro_running(when or self.when)

    def _add_metro(self, G):
        st = self.stations
        d, idx = self.tree.query(st[["x", "y"]].values)
        for (sid, row), dist, k in zip(st.iterrows(), d, idx):
            if dist > STATION_ACCESS_M:
                continue
            walk_node = self.ids[k]
            access = dist / WALK_SPEED_MPS
            for line in row.lines:
                p = ("metro", line, sid)
                G.add_edge(walk_node, p, cost=access + METRO_WAIT_S,
                           mode="board", wait=METRO_WAIT_S, access_m=dist)
                G.add_edge(p, walk_node, cost=access, mode="alight", access_m=dist)
            for a in row.lines:
                for b in row.lines:
                    if a != b:
                        G.add_edge(("metro", a, sid), ("metro", b, sid),
                                   cost=TRANSFER_S + METRO_WAIT_S, mode="transfer",
                                   wait=TRANSFER_S + METRO_WAIT_S)
        for line, seq in self.lines.items():
            for s1, s2 in zip(seq[:-1], seq[1:]):
                if s1 not in st.index or s2 not in st.index:
                    continue
                dist = float(np.hypot(st.at[s1, "x"] - st.at[s2, "x"],
                                      st.at[s1, "y"] - st.at[s2, "y"]))
                t = dist / METRO_SPEED_MPS + METRO_DWELL_S
                for a, b in ((s1, s2), (s2, s1)):
                    G.add_edge(("metro", line, a), ("metro", line, b),
                               cost=t, mode="ride", ride=t)
        return G

    # ── queries ────────────────────────────────────────────────────────
    def nearest(self, lon, lat):
        x, y = _TO_UTM.transform(lon, lat)
        d, k = self.tree.query([x, y])
        return self.ids[k], d

    def route(self, o, d, beta=EXPOSURE_BETA, metro=False, kind=None):
        use_metro = bool(metro and self.stations is not None and self.metro_available())
        key = (beta, use_metro)
        if key not in self._cache:
            G = self._walk_graph(beta)
            self._cache[key] = self._add_metro(G) if use_metro else G
        G = self._cache[key]
        path = nx.shortest_path(G, o, d, weight="cost")
        r = Route(kind or ("walk+metro" if metro else f"walk beta={beta}"), path)
        cur = None
        for a, b in zip(path[:-1], path[1:]):
            ed = G[a][b]
            m = ed["mode"]
            if m == "walk":
                r.edges.append(ed["eid"])
                if cur != "walk":
                    r.legs.append(["walk", ""])
            elif m == "board":
                r.walk_m += ed["access_m"]
                r.wait_s += ed["wait"]
                r.legs.append(["metro", f"{b[1]} from {self._name(b[2])}"])
            elif m == "transfer":
                r.wait_s += ed["wait"]
                r.legs.append(["metro", f"change to {b[1]} at {self._name(b[2])}"])
            elif m == "ride":
                r.ride_s += ed["ride"]
            elif m == "alight":
                r.walk_m += ed["access_m"]
                r.legs[-1][1] += f" to {self._name(a[2])}"
            cur = "walk" if m == "walk" else "metro"
        self._stats(r)
        return r

    def compare(self, o, d, beta=EXPOSURE_BETA, with_metro=True):
        out = [self.route(o, d, beta=0.0, kind="shortest walk"),
               self.route(o, d, beta=beta, kind="safer walk")]
        if with_metro and self.stations is not None and self.metro_available():
            r = self.route(o, d, beta=beta, metro=True, kind="safer walk + metro")
            if any(leg[0] == "metro" for leg in r.legs):
                out.append(r)
        return out

    def _name(self, sid):
        return self.stations.at[sid, "name"] if self.stations is not None else str(sid)

    def _stats(self, r):
        if not r.edges:
            r.mean_score = r.min_score = r.mean_confidence = np.nan
            return
        s = self.e.loc[r.edges]
        r.walk_m += float(s.length_m.sum())
        r.mean_score = float(np.average(s.score, weights=s.length_m))
        r.min_score = float(s.score.min())
        r.exposed_m = float(s.length_m[s.score < LOW_SCORE].sum())
        r.mean_confidence = float(np.average(s.confidence, weights=s.length_m))
