"""Download TLC monthly trip files and reduce them to the committed aggregates.

    python scripts/build_data.py --first 2022-01 --last 2026-08

Resumable: each month is written to data/interim/ and skipped on re-runs.
Raw files are deleted after aggregation (use --keep-raw to keep them).
At the end the monthly pieces are combined into data/processed/.
Needs about 1 GB of free disk at a time and roughly 30 GB of downloads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from congestion.etl import SERVICES, aggregate_file, file_name, months, url  # noqa: E402

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


def download(service: str, month: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / file_name(service, month)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".part")
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url(service, month), tmp)
            tmp.rename(dest)
            return dest
        except Exception as e:  # network hiccup: retry with back-off
            if attempt == 3:
                raise
            print(f"  retry {service} {month}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    return dest


def build_month(service: str, month: str, keep_raw: bool) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    stem = INTERIM / f"{service}_{month}"
    if Path(f"{stem}_audit.json").exists():
        return
    t0 = time.time()
    raw = download(service, month)
    zone_day, group_hour, audit = aggregate_file(raw, service, month, duckdb.connect())
    zone_day.to_parquet(f"{stem}_zone_day.parquet", index=False)
    group_hour.to_parquet(f"{stem}_group_hour.parquet", index=False)
    Path(f"{stem}_audit.json").write_text(json.dumps(audit))
    if not keep_raw:
        raw.unlink()
    print(f"{service} {month}: {audit['rows_kept']:,} trips kept of "
          f"{audit['rows_raw']:,} ({time.time() - t0:.0f}s)", flush=True)


def compact(df: pd.DataFrame) -> pd.DataFrame:
    """Smaller files for the repository: cents for money and minutes/miles
    sums, 32-bit counts, categorical labels."""
    for c in df.columns:
        if c in ("service", "pu_group"):
            df[c] = df[c].astype("category")
        elif c.endswith("_sum"):
            df[c] = df[c].round(2)
        elif df[c].dtype.kind in "iu":
            df[c] = df[c].astype("int32")
    return df


def combine() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for kind in ("zone_day", "group_hour"):
        parts = sorted(INTERIM.glob(f"*_{kind}.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df = df.sort_values(list(df.columns[:4])).reset_index(drop=True)
        df = compact(df)
        df.to_parquet(PROCESSED / f"{kind}.parquet", index=False, compression="zstd",
                      compression_level=19)
        print(f"{kind}: {len(df):,} rows")
    audit = pd.DataFrame([json.loads(p.read_text())
                          for p in sorted(INTERIM.glob("*_audit.json"))])
    audit.sort_values(["service", "month"]).to_csv(PROCESSED / "audit.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="2022-01")
    ap.add_argument("--last", default="2026-08")
    ap.add_argument("--services", nargs="+", default=list(SERVICES))
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()
    if not args.combine_only:
        for month in months(args.first, args.last):
            for service in args.services:
                build_month(service, month, args.keep_raw)
    combine()


if __name__ == "__main__":
    main()
