"""Pull ADS-B Exchange readsb-hist sample-day snapshots (5 s cadence) and keep
every target within a box around an airport. Output: one parquet of raw reports.

Usage: python src/fetch_adsbx.py 2025-09-01 12 18 KIAD
"""
import gzip, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

AIRPORTS = {"KIAD": (38.9445, -77.4558), "KDCA": (38.8521, -77.0377)}
BOX_DEG = (0.09, 0.11)  # ~10 km lat x ~9.5 km lon: surface plus final/initial climb
FIELDS = ["hex", "type", "flight", "r", "t", "category", "alt_baro", "alt_geom", "gs", "track",
          "true_heading", "baro_rate", "lat", "lon", "nic", "nac_p", "sil", "seen_pos", "rssi"]

def fetch(session, day, hhmmss, ap):
    url = f"https://samples.adsbexchange.com/readsb-hist/{day.replace('-', '/')}/{hhmmss}Z.json.gz"
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            raw = r.content
            try:
                d = json.loads(gzip.decompress(raw))
            except OSError:
                d = json.loads(raw)
            break
        except Exception:
            time.sleep(2 * (attempt + 1))
    else:
        print("failed", url, flush=True)
        return []
    lat0, lon0 = AIRPORTS[ap]
    out = []
    for a in d["aircraft"]:
        if "lat" not in a or abs(a["lat"] - lat0) > BOX_DEG[0] or abs(a["lon"] - lon0) > BOX_DEG[1]:
            continue
        row = {k: a.get(k) for k in FIELDS}
        row["now"] = d["now"]
        out.append(row)
    return out

def main(day, h0, h1, ap):
    stamps = [f"{h:02d}{m:02d}{s:02d}" for h in range(h0, h1) for m in range(60) for s in range(0, 60, 5)]
    rows, s = [], requests.Session()
    with ThreadPoolExecutor(8) as ex:
        for i, chunk in enumerate(ex.map(lambda t: fetch(s, day, t, ap), stamps)):
            rows.extend(chunk)
            if i % 360 == 0:
                print(f"{i}/{len(stamps)} snapshots, {len(rows)} reports", flush=True)
    df = pd.DataFrame(rows)
    df["alt_baro"] = df["alt_baro"].astype(str)
    for c in ["flight", "r", "t", "category", "type", "hex"]:
        df[c] = df[c].astype("string").str.strip()
    out = Path("data/raw") / f"{ap}_{day}_{h0:02d}-{h1:02d}Z.parquet"
    df.to_parquet(out)
    print("wrote", out, len(df), "reports,", df.hex.nunique(), "targets", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
