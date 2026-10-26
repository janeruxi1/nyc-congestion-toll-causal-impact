"""Evidence for the CBD zone list in src/congestion/zones.py.

For January-March 2025 (Uber/Lyft and yellow taxis), computes the share of
trips that start and end in the same zone and carry the CBD fee. Zones in
the toll zone show a high share; zones outside show (almost) none; zones on
the 60th Street edge sit in between. Writes data/reference/cbd_zone_evidence.csv.

    python scripts/derive_cbd_zones.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from congestion.etl import file_name, url  # noqa: E402
from congestion.zones import zone_group  # noqa: E402

RAW = ROOT / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    parts = []
    for service in ("hvfhv", "yellow"):
        for month in ("2025-01", "2025-02", "2025-03"):
            path = RAW / file_name(service, month)
            fetched = not path.exists()
            if fetched:
                urllib.request.urlretrieve(url(service, month), path)
            parts.append(con.execute(f"""
                SELECT '{service}' AS service, PULocationID AS zone, count(*) AS trips,
                       count(*) FILTER (WHERE coalesce(cbd_congestion_fee, 0) > 0) AS fee_trips
                FROM read_parquet('{path}')
                WHERE PULocationID = DOLocationID AND PULocationID BETWEEN 1 AND 263
                GROUP BY ALL
            """).df())
            if fetched:
                path.unlink()
    df = pd.concat(parts).groupby(["zone", "service"], as_index=False)[["trips", "fee_trips"]].sum()
    wide = df.pivot(index="zone", columns="service", values=["trips", "fee_trips"])
    out = pd.DataFrame({
        "zone": wide.index,
        "hvfhv_trips": wide[("trips", "hvfhv")].fillna(0).astype(int).to_numpy(),
        "hvfhv_fee_share": (wide[("fee_trips", "hvfhv")] / wide[("trips", "hvfhv")]).round(3).to_numpy(),
        "yellow_trips": wide[("trips", "yellow")].fillna(0).astype(int).to_numpy(),
        "yellow_fee_share": (wide[("fee_trips", "yellow")] / wide[("trips", "yellow")]).round(3).to_numpy(),
    })
    out["group"] = out["zone"].map(zone_group)
    lookup = pd.read_csv(ROOT / "data" / "reference" / "taxi_zone_lookup.csv")
    out = out.merge(lookup[["LocationID", "Borough", "Zone"]], left_on="zone",
                    right_on="LocationID").drop(columns="LocationID")
    out.sort_values("hvfhv_fee_share", ascending=False).to_csv(
        ROOT / "data" / "reference" / "cbd_zone_evidence.csv", index=False)
    print(out.groupby("group")[["hvfhv_fee_share", "yellow_fee_share"]].describe().T.round(3))


if __name__ == "__main__":
    main()
