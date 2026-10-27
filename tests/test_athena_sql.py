"""The Athena (Trino) queries must reproduce the DuckDB ETL exactly.

Each query is translated to DuckDB with sqlglot and run on the same test
files as tests/test_etl_zones.py, through a view that behaves like the Athena
table: declared columns only, read by name, NULL where a month's file lacks
the column.
"""
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import sqlglot

from congestion.athena import RAW_COLUMNS, audit_sql, group_hour_sql, raw_key, raw_table_ddl, zone_day_sql
from congestion.etl import aggregate_file


def athena_like_view(con, service: str, path: Path, month: str) -> str:
    """A DuckDB view with the Athena table's columns over one monthly file."""
    present = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
    cols = [f'CAST("{c}" AS {t}) AS "{c}"' if c in present else f'CAST(NULL AS {t}) AS "{c}"'
            for c, t in RAW_COLUMNS[service]]
    name = f"raw_{service}"
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT {', '.join(cols)}, '{month}' AS month "
                f"FROM read_parquet('{path}')")
    return name


def run_trino(con, sql: str) -> pd.DataFrame:
    return con.execute(sqlglot.transpile(sql, read="trino", write="duckdb")[0]).df()


def same(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> None:
    a = a.sort_values(keys).reset_index(drop=True)
    b = b[a.columns].sort_values(keys).reset_index(drop=True)
    for c in a.columns:
        if a[c].dtype.kind in "fiu" or b[c].dtype.kind in "fiu":
            pd.testing.assert_series_equal(a[c].astype(float), b[c].astype(float), check_names=False)
        else:
            assert a[c].astype(str).tolist() == b[c].astype(str).tolist(), c


@pytest.mark.parametrize("fixture,service,month", [
    ("raw_hvfhv", "hvfhv", "2025-01"),
    ("raw_yellow_2022", "yellow", "2022-03"),
])
def test_athena_queries_match_duckdb_etl(request, fixture, service, month):
    path = request.getfixturevalue(fixture)
    zd, gh, audit = aggregate_file(path, service, month, duckdb.connect())
    con = duckdb.connect()
    table = athena_like_view(con, service, path, month)

    same(zd, run_trino(con, zone_day_sql(service, table)), ["date", "pu_zone"])
    same(gh, run_trino(con, group_hour_sql(service, table)), ["date", "hour", "pu_group"])
    got = run_trino(con, audit_sql(service, table)).iloc[0]
    for k in ("rows_raw", "rows_in_month", "rows_kept", "fee_trips"):
        assert got[k] == audit[k], k


def test_peak_hours_use_iso_weekdays(tmp_path):
    """Trino counts Monday = 1 ... Sunday = 7; DuckDB's dayofweek has Sunday = 0.
    One trip at 07:00 on each day of a week: only the five weekdays are peak."""
    t0 = pd.Timestamp("2025-01-06 07:00")  # a Monday
    df = pd.DataFrame({
        "tpep_pickup_datetime": [t0 + pd.Timedelta(days=i) for i in range(7)],
        "tpep_dropoff_datetime": [t0 + pd.Timedelta(days=i, minutes=10) for i in range(7)],
        "PULocationID": [161] * 7, "DOLocationID": [161] * 7,
        "trip_distance": [1.0] * 7, "fare_amount": [10.0] * 7, "tip_amount": [1.0] * 7,
        "cbd_congestion_fee": [0.75] * 7,
    })
    path = tmp_path / "yellow_tripdata_2025-01.parquet"
    df.to_parquet(path, index=False)
    con = duckdb.connect()
    zd = run_trino(con, zone_day_sql("yellow", athena_like_view(con, "yellow", path, "2025-01")))
    assert zd.sort_values("date")["trips_peak"].tolist() == [1, 1, 1, 1, 1, 0, 0]


def test_raw_table_ddl_layout():
    ddl = raw_table_ddl("hvfhv", "tlc", "my-bucket")
    assert "LOCATION 's3://my-bucket/raw/hvfhv/'" in ddl
    assert "'storage.location.template' = 's3://my-bucket/raw/hvfhv/month=${month}/'" in ddl
    assert "`PULocationID` bigint" in ddl            # INT64 before Feb 2023, INT32 after
    assert raw_key("hvfhv", "2025-01") == "raw/hvfhv/month=2025-01/fhvhv_tripdata_2025-01.parquet"
    sqlglot.parse_one(zone_day_sql("hvfhv", "tlc.raw_hvfhv"), read="trino")  # valid Trino


def test_reconciliation_flags_every_kind_of_difference():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cloud_run", Path(__file__).resolve().parents[1] / "scripts" / "cloud_run.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)          # boto3 is imported lazily, so not needed here
    from congestion.etl import ZONE_DAY_COLUMNS

    base = {c: 0 for c in ZONE_DAY_COLUMNS}
    base.update(service="hvfhv", trips=100, fare_sum=1234.56, driver_pay_sum=999.99)
    local = pd.DataFrame([{**base, "date": pd.Timestamp(f"2025-0{m}-01"), "pu_zone": z}
                          for m in (1, 2, 3, 4) for z in (161, 74)])
    cloud = local.copy()
    cloud["date"] = cloud["date"].dt.date                    # Athena UNLOAD writes DATE
    cloud.loc[0, "fare_sum"] += 0.004                        # rounding: still a match
    cloud.loc[2, "trips"] += 1                               # Feb: a count differs
    cloud.loc[5, "driver_pay_sum"] = None                    # Mar: NULL on one side
    cloud = cloud.drop(index=7)                              # Apr: a row is missing
    rec = cr.reconcile(cloud, local, "zone_day").set_index("month")
    assert rec["match"].tolist() == [True, False, False, False]
    assert rec.loc["2025-02", "max_count_diff"] == 1
    assert rec.loc["2025-04", "rows_only_committed"] == 1
