"""Download KIAD runways, taxiways, aprons and hold-short positions from OpenStreetMap (Overpass)."""
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
Q = """[out:json][timeout:90];
(way["aeroway"~"runway|taxiway|apron"](38.90,-77.50,38.99,-77.41);
 node["aeroway"="holding_position"](38.90,-77.50,38.99,-77.41);
 way["aeroway"="holding_position"](38.90,-77.50,38.99,-77.41););
out geom tags;"""

if __name__ == "__main__":
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": Q},
                      headers={"User-Agent": "surface-safety-nowcast research"}, timeout=120)
    r.raise_for_status()
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    json.dump(r.json(), open(ROOT / "data/raw/kiad_osm.json", "w"))
    print(len(r.json()["elements"]), "OSM elements")
