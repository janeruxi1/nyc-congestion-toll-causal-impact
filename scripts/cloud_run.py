"""Run the TLC aggregation on AWS (S3 + Athena) and check it against the laptop build.

    python scripts/cloud_run.py all --stack tlc-athena      # copy, tables, build, compare
    python scripts/cloud_run.py compare                      # re-check downloaded results
    python scripts/cloud_run.py teardown --stack tlc-athena  # empty the bucket, delete the stack

Steps (each can be run on its own):
  copy      raw monthly files -> s3://<bucket>/raw/<service>/month=YYYY-MM/
            (server-side copy from the public nyc-tlc bucket; falls back to
            streaming from the TLC download site). Skips files already there.
  tables    one external table per service, partitioned by month
  build     zone_day, group_hour and audit for every service via UNLOAD,
            downloaded to data/cloud/ (not committed)
  compare   cloud output vs data/processed/, row by row
            -> reports/cloud/reconciliation.csv and reports/cloud/cloud_run.json
  sql       write the generated SQL to cloud/sql/ for reading (--check: verify
            the files are current; run in CI, needs no AWS account)

Setup, permissions and costs: cloud/README.md.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from congestion.athena import QUERIES, raw_key, raw_table, raw_table_ddl, unload_sql  # noqa: E402
from congestion.etl import GROUP_HOUR_COLUMNS, SERVICES, ZONE_DAY_COLUMNS, file_name, months, url  # noqa: E402

CLOUD = ROOT / "data" / "cloud"
PROCESSED = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "cloud"
RUN_RECORD = REPORT / "cloud_run.json"
PUBLIC_BUCKET, PUBLIC_PREFIX = "nyc-tlc", "trip data/"
PRICE_PER_TB = 5.0            # Athena, us-east-1, per TB scanned (10 MB minimum per query)
MIN_BILLED = 10 * 1024 ** 2
KEYS = {"zone_day": ["service", "date", "pu_zone"],
        "group_hour": ["service", "date", "hour", "pu_group"],
        "audit": ["service", "month"]}
COLUMNS = {"zone_day": ZONE_DAY_COLUMNS, "group_hour": GROUP_HOUR_COLUMNS,
           "audit": ["service", "month", "rows_raw", "rows_in_month", "rows_kept", "fee_trips"]}


# --------------------------------------------------------------------- AWS
def session(region: str):
    import boto3  # only needed for the cloud steps

    return boto3.session.Session(region_name=region)


def stack_outputs(sess, stack: str) -> dict:
    out = sess.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    return {o["OutputKey"]: o["OutputValue"] for o in out}


def run_query(sess, sql: str, workgroup: str, label: str) -> dict:
    athena = sess.client("athena")
    qid = athena.start_query_execution(QueryString=sql, WorkGroup=workgroup)["QueryExecutionId"]
    while True:
        q = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = q["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(2)
    if state != "SUCCEEDED":
        raise RuntimeError(f"{label}: {state}: {q['Status'].get('StateChangeReason', '')}")
    st = q.get("Statistics", {})
    scanned = int(st.get("DataScannedInBytes", 0))
    rec = {"query": label, "id": qid, "scanned_bytes": scanned,
           "billed_bytes": max(scanned, MIN_BILLED) if scanned else 0,
           "engine_seconds": st.get("EngineExecutionTimeInMillis", 0) / 1000,
           "total_seconds": st.get("TotalExecutionTimeInMillis", 0) / 1000}
    print(f"  {label}: {scanned / 1e9:.2f} GB scanned, {rec['total_seconds']:.0f}s", flush=True)
    return rec


def list_keys(s3, bucket: str, prefix: str) -> list[dict]:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += page.get("Contents", [])
    return keys


def delete_prefix(s3, bucket: str, prefix: str) -> None:
    keys = [{"Key": o["Key"]} for o in list_keys(s3, bucket, prefix)]
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": keys[i:i + 1000]})


# --------------------------------------------------------------------- steps
def copy(sess, out: dict, first: str, last: str) -> dict:
    from botocore.exceptions import ClientError

    s3, bucket = sess.client("s3"), out["BucketName"]
    have = {o["Key"] for o in list_keys(s3, bucket, "raw/")}

    def one(job):
        service, month = job
        key = raw_key(service, month)
        if key in have:
            return "already there", 0
        name = file_name(service, month)
        try:
            s3.copy({"Bucket": PUBLIC_BUCKET, "Key": PUBLIC_PREFIX + name}, bucket, key)
            source = "nyc-tlc bucket"
        except ClientError:   # no access to the public bucket: stream from the TLC site
            with urllib.request.urlopen(url(service, month), timeout=300) as r:
                s3.upload_fileobj(r, bucket, key)
            source = "TLC download site"
        size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        print(f"  {name}: {size / 1e6:.0f} MB from the {source}", flush=True)
        return source, size

    jobs = [(s, m) for s in SERVICES for m in months(first, last)]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(one, jobs))
    sources = pd.Series([r[0] for r in results]).value_counts().to_dict()
    total = sum(o["Size"] for o in list_keys(s3, bucket, "raw/"))
    print(f"copy: {len(jobs)} files, {total / 1e9:.1f} GB in the bucket {sources}")
    return {"files": len(jobs), "raw_bytes": total, "sources": sources}


def tables(sess, out: dict, first: str, last: str) -> list[dict]:
    db, wg = out["Database"], out["WorkGroup"]
    recs = []
    for s in SERVICES:
        run_query(sess, f"DROP TABLE IF EXISTS {db}.{raw_table(s)}", wg, f"drop {s}")
        recs.append(run_query(sess, raw_table_ddl(s, db, out["BucketName"], first, last), wg, f"create {s}"))
    return recs


def read_parquet_prefix(s3, bucket: str, prefix: str) -> pd.DataFrame:
    parts = []
    for o in list_keys(s3, bucket, prefix):
        body = s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()
        if body:
            parts.append(pd.read_parquet(io.BytesIO(body)))
    return pd.concat(parts, ignore_index=True)


def build(sess, out: dict) -> list[dict]:
    s3, bucket, db, wg = sess.client("s3"), out["BucketName"], out["Database"], out["WorkGroup"]
    CLOUD.mkdir(parents=True, exist_ok=True)
    recs = []
    for kind, make_sql in QUERIES.items():
        frames = []
        for s in SERVICES:
            prefix = f"out/{kind}/service={s}/"
            delete_prefix(s3, bucket, prefix)
            sql = unload_sql(make_sql(s, f"{db}.{raw_table(s)}"), f"s3://{bucket}/{prefix}")
            recs.append(run_query(sess, sql, wg, f"{kind} {s}"))
            frames.append(read_parquet_prefix(s3, bucket, prefix))
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(CLOUD / f"{kind}.parquet", index=False)
        print(f"{kind}: {len(df):,} rows -> data/cloud/{kind}.parquet")
    return recs


# --------------------------------------------------------------------- compare
def _normalise(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df[COLUMNS[kind]].copy()
    df["service"] = df["service"].astype(str)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"])
    if "month" in df:
        df["month"] = df["month"].astype(str)
    if "pu_group" in df:
        df["pu_group"] = df["pu_group"].astype(str)
    return df


def reconcile(cloud: pd.DataFrame, local: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Per service x month: same rows, identical counts, sums within rounding.

    The committed aggregates store money/miles/minutes sums rounded to cents
    (scripts/build_data.compact), so sums may differ by up to 0.005 plus
    floating-point summation order; 0.011 allows for both."""
    keys = KEYS[kind]
    c, loc = _normalise(cloud, kind), _normalise(local, kind)
    m = c.merge(loc, on=keys, how="outer", suffixes=("_cloud", "_local"), indicator=True)
    m["month"] = m["month"] if kind == "audit" else m["date"].dt.strftime("%Y-%m")
    values = [col for col in COLUMNS[kind] if col not in keys]
    counts = [v for v in values if not v.endswith("_sum")]
    sums = [v for v in values if v.endswith("_sum")]
    both = m["_merge"] == "both"
    for v in counts:
        m[f"d_{v}"] = (m[f"{v}_cloud"].fillna(0) - m[f"{v}_local"].fillna(0)).abs().where(both)
    for v in sums:
        a, b = m[f"{v}_cloud"], m[f"{v}_local"]
        diff = (a.fillna(0) - b.fillna(0)).abs()
        diff[a.isna() != b.isna()] = float("inf")       # NULL on one side only
        m[f"d_{v}"] = diff.where(both)
    rows = []
    for (service, month), g in m.groupby(["service", "month"]):
        gb = g[g["_merge"] == "both"]
        max_count = max((gb[f"d_{v}"].max() for v in counts), default=0)
        max_sum = max((gb[f"d_{v}"].max() for v in sums), default=0)
        r = {"table": kind, "service": service, "month": month,
             "rows_cloud": int((g["_merge"] != "right_only").sum()),
             "rows_committed": int((g["_merge"] != "left_only").sum()),
             "rows_only_cloud": int((g["_merge"] == "left_only").sum()),
             "rows_only_committed": int((g["_merge"] == "right_only").sum()),
             "max_count_diff": float(max_count if pd.notna(max_count) else 0),
             "max_sum_diff": float(max_sum if pd.notna(max_sum) else 0)}
        r["match"] = (r["rows_only_cloud"] == 0 and r["rows_only_committed"] == 0
                      and r["max_count_diff"] == 0 and r["max_sum_diff"] <= 0.011)
        rows.append(r)
    return pd.DataFrame(rows)


def compare() -> dict:
    local = {"zone_day": pd.read_parquet(PROCESSED / "zone_day.parquet"),
             "group_hour": pd.read_parquet(PROCESSED / "group_hour.parquet"),
             "audit": pd.read_csv(PROCESSED / "audit.csv", dtype={"month": str})}
    rec = pd.concat([reconcile(pd.read_parquet(CLOUD / f"{k}.parquet"), local[k], k) for k in KEYS],
                    ignore_index=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    rec.to_csv(REPORT / "reconciliation.csv", index=False)
    summary = {
        "service_months_checked": int(len(rec)),
        "service_months_matching": int(rec["match"].sum()),
        "all_match": bool(rec["match"].all()),
        "zone_day_rows": int(rec.loc[rec["table"] == "zone_day", "rows_cloud"].sum()),
        "group_hour_rows": int(rec.loc[rec["table"] == "group_hour", "rows_cloud"].sum()),
        "trips_kept": int(pd.read_parquet(CLOUD / "audit.parquet")["rows_kept"].sum()),
        "mismatches": rec.loc[~rec["match"], ["table", "service", "month"]].to_dict("records"),
    }
    print(f"compare: {summary['service_months_matching']} of {summary['service_months_checked']} "
          f"table x service x month checks match")
    return summary


# --------------------------------------------------------------------- sql files
SQL_DIR = ROOT / "cloud" / "sql"


def sql_files() -> dict[str, str]:
    """Every statement the pipeline runs, with placeholders for the account's names."""
    files = {}
    for s in SERVICES:
        files[f"00_create_raw_{s}.sql"] = raw_table_ddl(s, "tlc", "<your-bucket>")
    for i, (kind, make_sql) in enumerate(QUERIES.items(), start=1):
        for s in SERVICES:
            files[f"{i:02d}_{kind}_{s}.sql"] = unload_sql(make_sql(s, f"tlc.{raw_table(s)}"),
                                                         f"s3://<your-bucket>/out/{kind}/service={s}/")
    header = "-- Generated by scripts/cloud_run.py sql from src/congestion/athena.py. Do not edit.\n"
    return {k: header + v + ";\n" for k, v in files.items()}


def write_sql(check: bool) -> None:
    files = sql_files()
    if check:
        stale = [k for k, v in files.items() if not (SQL_DIR / k).exists() or (SQL_DIR / k).read_text() != v]
        extra = sorted({p.name for p in SQL_DIR.glob("*.sql")} - set(files))
        if stale or extra:
            sys.exit(f"cloud/sql is out of date ({stale + extra}); run: python scripts/cloud_run.py sql")
        print(f"cloud/sql: all {len(files)} files are current.")
        return
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        (SQL_DIR / k).write_text(v)
    print(f"Wrote {len(files)} files to cloud/sql/")


# --------------------------------------------------------------------- record
def write_record(update: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    rec = json.loads(RUN_RECORD.read_text()) if RUN_RECORD.exists() else {}
    rec.update(update)
    queries = rec.get("queries", [])
    billed = sum(q["billed_bytes"] for q in queries)
    rec["totals"] = {
        "queries": len(queries),
        "scanned_gb": round(sum(q["scanned_bytes"] for q in queries) / 1e9, 2),
        "athena_cost_usd": round(billed / 1024 ** 4 * PRICE_PER_TB, 3),
        "query_minutes": round(sum(q["total_seconds"] for q in queries) / 60, 1),
    }
    RUN_RECORD.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(f"Saved {RUN_RECORD.relative_to(ROOT)}")


def teardown(sess, stack: str) -> None:
    out = stack_outputs(sess, stack)
    s3 = sess.resource("s3")
    s3.Bucket(out["BucketName"]).objects.all().delete()
    cf = sess.client("cloudformation")
    cf.delete_stack(StackName=stack)
    cf.get_waiter("stack_delete_complete").wait(StackName=stack)
    print(f"Deleted stack {stack} and everything in it")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "copy", "tables", "build", "compare", "teardown", "sql"])
    ap.add_argument("--check", action="store_true", help="with sql: fail if cloud/sql/ is stale")
    ap.add_argument("--stack", default="tlc-athena")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--first", default="2022-01")
    ap.add_argument("--last", default="2026-08")
    args = ap.parse_args()

    if args.step == "sql":
        write_sql(args.check)
        return
    if args.step == "compare":
        write_record({"compare": compare()})
        return
    sess = session(args.region)
    if args.step == "teardown":
        teardown(sess, args.stack)
        return
    out = stack_outputs(sess, args.stack)
    t0 = time.time()
    update: dict = {"stack": args.stack, "region": args.region, "months": f"{args.first} to {args.last}",
                    "engine": "Athena engine version 3",
                    "price_per_tb_usd": PRICE_PER_TB}
    if args.step in ("all", "copy"):
        update["copy"] = copy(sess, out, args.first, args.last)
    if args.step in ("all", "tables"):
        tables(sess, out, args.first, args.last)
    if args.step in ("all", "build"):
        update["queries"] = build(sess, out)
        update["run_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if args.step in ("all", "compare"):
        update["compare"] = compare()
    update["wall_minutes_last_step"] = round((time.time() - t0) / 60, 1)
    write_record(update)


if __name__ == "__main__":
    main()
