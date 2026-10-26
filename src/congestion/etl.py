"""Monthly TLC trip files -> small daily aggregates.

One monthly file holds up to 20 million trips. Each is reduced with DuckDB to
two tables and then discarded, so the raw data never has to sit on disk all
at once and the repository only stores aggregates:

* ``zone_day``  - one row per service x date x pickup zone (about 1.3M rows
  for 2022-2026, all services)
* ``group_hour`` - one row per service x date x hour x pickup group, used for
  the forecasting work

Services: ``yellow`` (yellow taxis), ``green`` (green "boro" taxis) and
``hvfhv`` (high-volume for-hire vehicles: Uber, Lyft).

Cleaning rules (counted per month in the audit table):
* pickup time inside the file's month (files contain a few stray dates)
* trip duration between 1 minute and 4 hours
* distance between 0 and 100 miles, fare not negative
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

from .zones import CBD_ZONES, group_case_sql

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
SERVICES = ("yellow", "green", "hvfhv")
FILE_PREFIX = {"yellow": "yellow", "green": "green", "hvfhv": "fhvhv"}

# Car toll hours (the per-trip taxi/FHV fee applies at all hours):
# weekdays 05:00-21:00, weekends 09:00-21:00.
PEAK_SQL = ("(CASE WHEN dayofweek(pickup) IN (0, 6) "
            "THEN hour(pickup) BETWEEN 9 AND 20 "
            "ELSE hour(pickup) BETWEEN 5 AND 20 END)")

ZONE_DAY_COLUMNS = [
    "service", "date", "pu_zone", "trips", "trips_peak", "trips_to_cbd",
    "fee_trips", "fee_sum", "fare_sum", "tips_sum", "driver_pay_sum",
    "miles_sum", "minutes_sum", "wait_trips", "wait_min_sum",
    "shared_requests", "shared_matches",
]
GROUP_HOUR_COLUMNS = [
    "service", "date", "hour", "pu_group", "trips", "trips_to_cbd", "fee_trips",
    "fare_sum", "driver_pay_sum", "miles_sum", "minutes_sum",
]


def file_name(service: str, month: str) -> str:
    return f"{FILE_PREFIX[service]}_tripdata_{month}.parquet"


def url(service: str, month: str) -> str:
    return f"{BASE_URL}/{file_name(service, month)}"


def _columns(con: duckdb.DuckDBPyConnection, path: str) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    return {r[0].lower(): r[0] for r in rows}


def _col(cols: dict, name: str, default: str = "NULL") -> str:
    """Quoted column if the file has it (any case), else a default literal."""
    real = cols.get(name.lower())
    return f'"{real}"' if real else default


def standard_trips_sql(cols: dict, service: str, path: str) -> str:
    """SELECT that maps one raw file onto a common trip schema."""
    if service == "hvfhv":
        pickup, dropoff = _col(cols, "pickup_datetime"), _col(cols, "dropoff_datetime")
        fare, tips = _col(cols, "base_passenger_fare"), _col(cols, "tips", "0")
        miles = _col(cols, "trip_miles")
        driver_pay = _col(cols, "driver_pay")
        request = _col(cols, "request_datetime")
        shared_req = f"({_col(cols, 'shared_request_flag', 'NULL')} = 'Y')"
        shared_match = f"({_col(cols, 'shared_match_flag', 'NULL')} = 'Y')"
    else:
        p = "tpep" if service == "yellow" else "lpep"
        pickup, dropoff = _col(cols, f"{p}_pickup_datetime"), _col(cols, f"{p}_dropoff_datetime")
        fare, tips = _col(cols, "fare_amount"), _col(cols, "tip_amount", "0")
        miles = _col(cols, "trip_distance")
        driver_pay = request = "NULL"
        shared_req = shared_match = "NULL"
    fee = _col(cols, "cbd_congestion_fee", "0")
    return f"""
        SELECT
            CAST({pickup} AS TIMESTAMP)                          AS pickup,
            CAST({dropoff} AS TIMESTAMP)                         AS dropoff,
            CAST({request} AS TIMESTAMP)                         AS request,
            CAST("PULocationID" AS INTEGER)                      AS pu_zone,
            CAST("DOLocationID" AS INTEGER)                      AS do_zone,
            CAST({fare} AS DOUBLE)                               AS fare,
            CAST(coalesce({tips}, 0) AS DOUBLE)                  AS tips,
            CAST({driver_pay} AS DOUBLE)                         AS driver_pay,
            CAST({miles} AS DOUBLE)                              AS miles,
            CAST(coalesce({fee}, 0) AS DOUBLE)                   AS fee,
            {shared_req}                                         AS shared_req,
            {shared_match}                                       AS shared_match
        FROM read_parquet('{path}')
    """


def aggregate_file(path: str | Path, service: str, month: str,
                   con: duckdb.DuckDBPyConnection | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reduce one monthly file to (zone_day, group_hour, audit)."""
    con = con or duckdb.connect()
    path = str(path)
    cols = _columns(con, path)
    start = dt.date.fromisoformat(f"{month}-01")
    end = (start + dt.timedelta(days=32)).replace(day=1)
    cbd = ", ".join(str(z) for z in sorted(CBD_ZONES))

    con.execute(f"CREATE OR REPLACE TEMP VIEW raw AS {standard_trips_sql(cols, service, path)}")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE clean AS
        SELECT *,
               date_diff('second', pickup, dropoff) / 60.0 AS minutes,
               CASE WHEN request IS NOT NULL
                         AND date_diff('second', request, pickup) BETWEEN 0 AND 3600
                    THEN date_diff('second', request, pickup) / 60.0 END AS wait_min
        FROM raw
        WHERE pickup >= DATE '{start}' AND pickup < DATE '{end}'
          AND date_diff('second', pickup, dropoff) BETWEEN 60 AND 4 * 3600
          AND miles >= 0 AND miles <= 100
          AND (fare IS NULL OR fare >= 0)
    """)
    audit = {
        "service": service, "month": month,
        "rows_raw": con.execute("SELECT count(*) FROM raw").fetchall()[0][0],
        "rows_in_month": con.execute(
            f"SELECT count(*) FROM raw WHERE pickup >= DATE '{start}' "
            f"AND pickup < DATE '{end}'").fetchall()[0][0],
        "rows_kept": con.execute("SELECT count(*) FROM clean").fetchall()[0][0],
        "fee_trips": con.execute("SELECT count(*) FROM clean WHERE fee > 0").fetchall()[0][0],
    }

    zone_day = con.execute(f"""
        SELECT '{service}' AS service, CAST(pickup AS DATE) AS date, pu_zone,
               count(*)                                       AS trips,
               count(*) FILTER (WHERE {PEAK_SQL})             AS trips_peak,
               count(*) FILTER (WHERE do_zone IN ({cbd}))     AS trips_to_cbd,
               count(*) FILTER (WHERE fee > 0)                AS fee_trips,
               sum(fee)                                       AS fee_sum,
               sum(fare)                                      AS fare_sum,
               sum(tips)                                      AS tips_sum,
               sum(driver_pay)                                AS driver_pay_sum,
               sum(miles)                                     AS miles_sum,
               sum(minutes)                                   AS minutes_sum,
               count(wait_min)                                AS wait_trips,
               sum(wait_min)                                  AS wait_min_sum,
               count(*) FILTER (WHERE shared_req)             AS shared_requests,
               count(*) FILTER (WHERE shared_match)           AS shared_matches
        FROM clean
        WHERE pu_zone BETWEEN 1 AND 263
        GROUP BY ALL ORDER BY ALL
    """).df()

    group_hour = con.execute(f"""
        SELECT '{service}' AS service, CAST(pickup AS DATE) AS date,
               hour(pickup) AS hour, {group_case_sql('pu_zone')} AS pu_group,
               count(*)                                       AS trips,
               count(*) FILTER (WHERE do_zone IN ({cbd}))     AS trips_to_cbd,
               count(*) FILTER (WHERE fee > 0)                AS fee_trips,
               sum(fare)                                      AS fare_sum,
               sum(driver_pay)                                AS driver_pay_sum,
               sum(miles)                                     AS miles_sum,
               sum(minutes)                                   AS minutes_sum
        FROM clean
        GROUP BY ALL ORDER BY ALL
    """).df()
    con.execute("DROP TABLE clean")
    return zone_day[ZONE_DAY_COLUMNS], group_hour[GROUP_HOUR_COLUMNS], audit


def months(first: str, last: str) -> list[str]:
    """'2022-01', '2022-02', ... up to and including `last`."""
    y, m = map(int, first.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out
