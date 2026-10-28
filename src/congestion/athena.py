"""The ETL of `etl.py`, written for Amazon Athena (Trino SQL).

`etl.aggregate_file` reduces one monthly file at a time on a laptop with
DuckDB. Here the same rules run as three queries per service over all 56
months at once, on the raw files stored in S3. The two pipelines must agree:
tests/test_athena_sql.py translates these queries to DuckDB and checks them
against `aggregate_file` on test files, and scripts/cloud_run.py compares
the cloud output with the committed aggregates row by row.

Schema drift in the raw files (checked on all 168 files' footers):
* PULocationID / DOLocationID are INT64 until January 2023 and INT32 after,
  so the tables declare BIGINT (Athena widens INT32 to BIGINT, not the
  reverse).
* `cbd_congestion_fee` only exists from January 2025. Athena reads Parquet
  columns by name, so earlier files return NULL, treated as 0 below.
* The shared-ride flags change from STRING to LARGE_STRING in February 2023;
  both are Parquet BYTE_ARRAY and read as `string` (Athena's table DDL is
  Hive syntax, where `varchar` needs a length).
"""
from __future__ import annotations

from .etl import FILE_PREFIX, GROUP_HOUR_COLUMNS, SERVICES, ZONE_DAY_COLUMNS
from .zones import CBD_ZONES, group_case_sql

# Columns the pipeline reads, with the Athena type that covers every month.
RAW_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "yellow": [
        ("tpep_pickup_datetime", "timestamp"), ("tpep_dropoff_datetime", "timestamp"),
        ("PULocationID", "bigint"), ("DOLocationID", "bigint"),
        ("fare_amount", "double"), ("tip_amount", "double"), ("trip_distance", "double"),
        ("cbd_congestion_fee", "double"),
    ],
    "green": [
        ("lpep_pickup_datetime", "timestamp"), ("lpep_dropoff_datetime", "timestamp"),
        ("PULocationID", "bigint"), ("DOLocationID", "bigint"),
        ("fare_amount", "double"), ("tip_amount", "double"), ("trip_distance", "double"),
        ("cbd_congestion_fee", "double"),
    ],
    "hvfhv": [
        ("pickup_datetime", "timestamp"), ("dropoff_datetime", "timestamp"),
        ("request_datetime", "timestamp"),
        ("PULocationID", "bigint"), ("DOLocationID", "bigint"),
        ("base_passenger_fare", "double"), ("tips", "double"), ("trip_miles", "double"),
        ("driver_pay", "double"), ("shared_request_flag", "string"),
        ("shared_match_flag", "string"), ("cbd_congestion_fee", "double"),
    ],
}

# Toll hours for cars, as in etl.PEAK_SQL. Trino's day_of_week is ISO
# (Monday = 1 ... Sunday = 7); DuckDB's dayofweek has Sunday = 0.
PEAK_SQL = ("(CASE WHEN day_of_week(pickup) IN (6, 7) "
            "THEN hour(pickup) BETWEEN 9 AND 20 "
            "ELSE hour(pickup) BETWEEN 5 AND 20 END)")


def raw_table(service: str) -> str:
    return f"raw_{service}"


def raw_prefix(service: str) -> str:
    """S3 key prefix of one service's raw files: raw/<service>/month=YYYY-MM/."""
    return f"raw/{service}/"


def raw_key(service: str, month: str) -> str:
    return f"{raw_prefix(service)}month={month}/{FILE_PREFIX[service]}_tripdata_{month}.parquet"


def raw_table_ddl(service: str, database: str, bucket: str,
                  first: str = "2022-01", last: str = "2026-08") -> str:
    """External table over the raw files, one partition per month.

    Partition projection computes the partitions from the key layout, so no
    crawler or MSCK REPAIR is needed and queries only list the months asked for.
    """
    cols = ",\n  ".join(f"`{c}` {t}" for c, t in RAW_COLUMNS[service])
    location = f"s3://{bucket}/{raw_prefix(service)}"
    return f"""CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{raw_table(service)} (
  {cols}
)
PARTITIONED BY (`month` string)
STORED AS PARQUET
LOCATION '{location}'
TBLPROPERTIES (
  'projection.enabled' = 'true',
  'projection.month.type' = 'date',
  'projection.month.format' = 'yyyy-MM',
  'projection.month.range' = '{first},{last}',
  'projection.month.interval' = '1',
  'projection.month.interval.unit' = 'MONTHS',
  'storage.location.template' = '{location}month=${{month}}/'
)"""


def _standard_trips(service: str, table: str) -> str:
    """Raw columns mapped onto the common trip schema of etl.standard_trips_sql."""
    if service == "hvfhv":
        pickup, dropoff, request = "pickup_datetime", "dropoff_datetime", "request_datetime"
        fare, tips, miles, pay = "base_passenger_fare", "tips", "trip_miles", "driver_pay"
        shared_req = "(shared_request_flag = 'Y')"
        shared_match = "(shared_match_flag = 'Y')"
    else:
        p = "tpep" if service == "yellow" else "lpep"
        pickup, dropoff, request = f"{p}_pickup_datetime", f"{p}_dropoff_datetime", "NULL"
        fare, tips, miles, pay = "fare_amount", "tip_amount", "trip_distance", "NULL"
        shared_req = shared_match = "CAST(NULL AS boolean)"
    cols = [
        ("month", None),
        (f"CAST({pickup} AS timestamp)", "pickup"),
        (f"CAST({dropoff} AS timestamp)", "dropoff"),
        (f"CAST({request} AS timestamp)", "request"),
        ("CAST(PULocationID AS integer)", "pu_zone"),
        ("CAST(DOLocationID AS integer)", "do_zone"),
        (f"CAST({fare} AS double)", "fare"),
        (f"CAST(coalesce({tips}, 0) AS double)", "tips"),
        (f"CAST({pay} AS double)", "driver_pay"),
        (f"CAST({miles} AS double)", "miles"),
        ("CAST(coalesce(cbd_congestion_fee, 0) AS double)", "fee"),
        (shared_req, "shared_req"),
        (shared_match, "shared_match"),
    ]
    width = max(len(e) for e, _ in cols)
    body = ",\n        ".join(e if a is None else f"{e:<{width}} AS {a}" for e, a in cols)
    return f"""SELECT
        {body}
    FROM {table}"""


def _month_start() -> str:
    return "date_parse(month, '%Y-%m')"


def _clean_cte(service: str, table: str) -> str:
    """WITH raw AS (...), clean AS (...): the cleaning rules of etl.aggregate_file."""
    start = _month_start()
    secs = "date_diff('second', pickup, dropoff)"
    wait = "date_diff('second', request, pickup)"
    return f"""WITH raw AS (
    {_standard_trips(service, table)}
), clean AS (
    SELECT *,
           {secs} / 6e1 AS minutes,
           CASE WHEN request IS NOT NULL AND {wait} BETWEEN 0 AND 3600
                THEN {wait} / 6e1 END AS wait_min
    FROM raw
    WHERE pickup >= {start} AND pickup < date_add('month', 1, {start})
      AND {secs} BETWEEN 60 AND 4 * 3600
      AND miles >= 0 AND miles <= 100
      AND (fare IS NULL OR fare >= 0)
)"""


def _cbd_list() -> str:
    return ", ".join(str(z) for z in sorted(CBD_ZONES))


def zone_day_sql(service: str, table: str) -> str:
    """One row per date x pickup zone (zones 1-263), as etl.ZONE_DAY_COLUMNS."""
    sql = f"""{_clean_cte(service, table)}
SELECT '{service}' AS service, CAST(pickup AS date) AS date, pu_zone,
       count(*)                                          AS trips,
       count(*) FILTER (WHERE {PEAK_SQL})                AS trips_peak,
       count(*) FILTER (WHERE do_zone IN ({_cbd_list()})) AS trips_to_cbd,
       count(*) FILTER (WHERE fee > 0)                   AS fee_trips,
       sum(fee)                                          AS fee_sum,
       sum(fare)                                         AS fare_sum,
       sum(tips)                                         AS tips_sum,
       sum(driver_pay)                                   AS driver_pay_sum,
       sum(miles)                                        AS miles_sum,
       sum(minutes)                                      AS minutes_sum,
       count(wait_min)                                   AS wait_trips,
       sum(wait_min)                                     AS wait_min_sum,
       count(*) FILTER (WHERE shared_req)                AS shared_requests,
       count(*) FILTER (WHERE shared_match)              AS shared_matches
FROM clean
WHERE pu_zone BETWEEN 1 AND 263
GROUP BY 1, 2, 3"""
    _check_columns(sql, ZONE_DAY_COLUMNS)
    return sql


def group_hour_sql(service: str, table: str) -> str:
    """One row per date x hour x pickup group, as etl.GROUP_HOUR_COLUMNS."""
    sql = f"""{_clean_cte(service, table)}
SELECT '{service}' AS service, CAST(pickup AS date) AS date, hour(pickup) AS hour,
       {group_case_sql('pu_zone')} AS pu_group,
       count(*)                                          AS trips,
       count(*) FILTER (WHERE do_zone IN ({_cbd_list()})) AS trips_to_cbd,
       count(*) FILTER (WHERE fee > 0)                   AS fee_trips,
       sum(fare)                                         AS fare_sum,
       sum(driver_pay)                                   AS driver_pay_sum,
       sum(miles)                                        AS miles_sum,
       sum(minutes)                                      AS minutes_sum
FROM clean
GROUP BY 1, 2, 3, 4"""
    _check_columns(sql, GROUP_HOUR_COLUMNS)
    return sql


def audit_sql(service: str, table: str) -> str:
    """Row counts per month at each cleaning step, as data/processed/audit.csv."""
    start = _month_start()
    secs = "date_diff('second', pickup, dropoff)"
    return f"""WITH raw AS (
    {_standard_trips(service, table)}
)
SELECT '{service}' AS service, month,
       count(*) AS rows_raw,
       count(*) FILTER (WHERE pickup >= {start} AND pickup < date_add('month', 1, {start})) AS rows_in_month,
       count(*) FILTER (WHERE pickup >= {start} AND pickup < date_add('month', 1, {start})
                          AND {secs} BETWEEN 60 AND 4 * 3600
                          AND miles >= 0 AND miles <= 100
                          AND (fare IS NULL OR fare >= 0)) AS rows_kept,
       count(*) FILTER (WHERE pickup >= {start} AND pickup < date_add('month', 1, {start})
                          AND {secs} BETWEEN 60 AND 4 * 3600
                          AND miles >= 0 AND miles <= 100
                          AND (fare IS NULL OR fare >= 0)
                          AND fee > 0) AS fee_trips
FROM raw
GROUP BY 1, 2"""


QUERIES = {"zone_day": zone_day_sql, "group_hour": group_hour_sql, "audit": audit_sql}


def unload_sql(select: str, s3_prefix: str) -> str:
    """Write a query's result to S3 as Parquet (the prefix must be empty)."""
    return f"UNLOAD (\n{select}\n)\nTO '{s3_prefix}'\nWITH (format = 'PARQUET', compression = 'SNAPPY')"


def _check_columns(sql: str, expected: list[str]) -> None:
    """Guard against the two pipelines' column lists drifting apart."""
    import sqlglot

    select = sqlglot.parse_one(sql, read="trino")
    got = [e.alias_or_name for e in select.expressions]
    if got != expected:
        raise ValueError(f"column mismatch: {got} != {expected}")


__all__ = ["RAW_COLUMNS", "SERVICES", "QUERIES", "raw_table", "raw_key", "raw_prefix",
           "raw_table_ddl", "zone_day_sql", "group_hour_sql", "audit_sql", "unload_sql"]
