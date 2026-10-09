"""Airport surface geometry for KIAD from OpenStreetMap: local metric frame, runways,
runway hold-short lines, and a taxiway graph for route distances."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = (38.9445, -77.4558)  # KIAD reference point
R_EARTH = 6371008.8
FT = 0.3048


def to_xy(lat, lon):
    """Equirectangular local frame in metres (x east, y north); <0.1 m error over 10 km."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = np.radians(lon - REF[1]) * R_EARTH * np.cos(np.radians(REF[0]))
    y = np.radians(lat - REF[0]) * R_EARTH
    return x, y


@dataclass
class Runway:
    name: str          # e.g. "01C/19C"
    a: np.ndarray      # end of the low-numbered direction's threshold (xy)
    b: np.ndarray
    width: float

    @property
    def length(self):
        return float(np.linalg.norm(self.b - self.a))

    @property
    def unit(self):
        return (self.b - self.a) / self.length

    @property
    def heading(self):
        """True heading (deg) travelling from a to b."""
        u = self.unit
        return float(np.degrees(np.arctan2(u[0], u[1])) % 360)

    def along_cross(self, x, y):
        """Distance along the centreline from a, and signed lateral offset (m)."""
        p = np.stack([np.asarray(x) - self.a[0], np.asarray(y) - self.a[1]], -1)
        u = self.unit
        along = p @ u
        cross = p @ np.array([-u[1], u[0]])
        return along, cross

    def ends(self):
        lo, hi = self.name.split("/")
        # the end whose heading matches travel direction a->b is the low-numbered end if heading ~ 10*number
        h = self.heading
        lo_num = int(lo[:2]) * 10
        if min(abs(h - lo_num), 360 - abs(h - lo_num)) < 45:
            return {lo: (self.a, h), hi: (self.b, (h + 180) % 360)}
        return {lo: (self.b, (h + 180) % 360), hi: (self.a, h)}


@dataclass
class HoldLine:
    id: int
    xy: np.ndarray
    runway: str        # runway this line protects
    side: float        # sign of lateral offset (which side of the runway)
    along: float       # position along that runway (m from a)


class Airport:
    def __init__(self, path=ROOT / "data/raw/kiad_osm.json"):
        els = json.load(open(path))["elements"]
        self.runways, self.holds, self.taxiways = {}, [], []
        for e in els:
            t = e.get("tags", {})
            if t.get("aeroway") == "runway":
                g = e["geometry"]
                x, y = to_xy([g[0]["lat"], g[-1]["lat"]], [g[0]["lon"], g[-1]["lon"]])
                self.runways[t["ref"]] = Runway(t["ref"], np.array([x[0], y[0]]), np.array([x[1], y[1]]),
                                                float(t.get("width", 46)))
            elif t.get("aeroway") == "taxiway":
                g = e["geometry"]
                x, y = to_xy([p["lat"] for p in g], [p["lon"] for p in g])
                self.taxiways.append((t.get("ref"), np.stack([x, y], 1)))
        for e in els:
            t = e.get("tags", {})
            if t.get("aeroway") != "holding_position" or t.get("holding_position:type", "runway") != "runway":
                continue
            if e["type"] == "node":
                lat, lon = e["lat"], e["lon"]
            else:
                lat = np.mean([p["lat"] for p in e["geometry"]]); lon = np.mean([p["lon"] for p in e["geometry"]])
            x, y = to_xy(lat, lon)
            best = None
            for rw in self.runways.values():
                al, cr = rw.along_cross(x, y)
                if -300 < al < rw.length + 300:
                    d = abs(float(cr))
                    if best is None or d < best[0]:
                        best = (d, rw, float(al), float(np.sign(cr)))
            if best and best[0] < 200:
                self.holds.append(HoldLine(e["id"], np.array([float(x), float(y)]), best[1].name, best[3], best[2]))
        self.graph = self._graph()

    def _graph(self):
        G = nx.Graph()
        key = lambda p: (round(float(p[0]), 1), round(float(p[1]), 1))
        for _, pts in self.taxiways:
            for p, q in zip(pts[:-1], pts[1:]):
                G.add_edge(key(p), key(q), w=float(np.linalg.norm(q - p)))
        # stitch near-coincident nodes (OSM ways that meet without sharing a node)
        nodes = np.array(list(G.nodes))
        for i, n in enumerate(nodes):
            d = np.linalg.norm(nodes - n, axis=1)
            for j in np.where((d > 0) & (d < 8))[0]:
                G.add_edge(tuple(n), tuple(nodes[j]), w=float(d[j]))
        self._nodes = np.array(list(G.nodes))
        return G

    def nearest_node(self, x, y):
        i = int(np.argmin(np.hypot(self._nodes[:, 0] - x, self._nodes[:, 1] - y)))
        return tuple(self._nodes[i]), float(np.hypot(*(self._nodes[i] - [x, y])))

    @lru_cache(maxsize=None)
    def _dist_from(self, node):
        return nx.single_source_dijkstra_path_length(self.graph, node, weight="w")

    def route_distance(self, x, y, target_xy):
        """Taxi-route distance (m) from a position to a target point over the taxiway graph."""
        n0, d0 = self.nearest_node(x, y)
        n1, d1 = self.nearest_node(*target_xy)
        d = self._dist_from(n1).get(n0)
        return np.nan if d is None else d + d0 + d1

    def runway_context(self, x, y):
        """Per-point nearest runway, |lateral offset|, along-distance, and whether inside the runway strip."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        best = np.full(len(x), np.inf); name = np.empty(len(x), object); along = np.zeros(len(x))
        for rw in self.runways.values():
            al, cr = rw.along_cross(x, y)
            outside = np.clip(-al, 0, None) + np.clip(al - rw.length, 0, None)
            d = np.hypot(np.abs(cr), outside)
            m = d < best
            best[m], name[m], along[m] = d[m], rw.name, al[m]
        return name, best, along

    def nearest_hold(self, x, y):
        H = np.array([h.xy for h in self.holds])
        d = np.hypot(H[:, 0][None] - np.atleast_1d(x)[:, None], H[:, 1][None] - np.atleast_1d(y)[:, None])
        i = d.argmin(1)
        return i, d[np.arange(len(i)), i]


if __name__ == "__main__":
    ap = Airport()
    for r in ap.runways.values():
        print(r.name, round(r.length), "m, hdg", round(r.heading), r.ends().keys())
    from collections import Counter
    print(len(ap.holds), "runway hold lines:", Counter(h.runway for h in ap.holds))
    print(ap.graph.number_of_nodes(), "taxi nodes,", nx.number_connected_components(ap.graph), "components; largest",
          len(max(nx.connected_components(ap.graph), key=len)))
