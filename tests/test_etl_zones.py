import duckdb
import pandas as pd

from congestion.etl import aggregate_file, months
from congestion.zones import BORDER_ZONES, CBD_ZONES, group_case_sql, zone_group


def test_zone_groups_are_disjoint_and_complete():
    assert not CBD_ZONES & BORDER_ZONES
    assert len(CBD_ZONES) == 38
    assert zone_group(161) == "cbd" and zone_group(237) == "border"
    assert zone_group(132) == "airport" and zone_group(74) == "manhattan_north"
    assert zone_group(61) == "outer" and zone_group(264) == "unknown"


def test_sql_grouping_matches_python():
    con = duckdb.connect()
    sql = group_case_sql("z")
    got = con.execute(f"SELECT z, {sql} FROM range(0, 267) t(z)").fetchall()
    assert all(g == zone_group(z) for z, g in got)


def test_aggregate_hvfhv_counts_and_filters(raw_hvfhv):
    zd, gh, audit = aggregate_file(raw_hvfhv, "hvfhv", "2025-01")
    assert audit["rows_raw"] == 8
    assert audit["rows_in_month"] == 7          # the 31 Dec trip is out
    assert audit["rows_kept"] == 6              # the 30-second trip is out
    by_zone = zd.groupby("pu_zone")[["trips", "trips_to_cbd", "fee_trips", "trips_peak"]].sum()
    assert by_zone.loc[161].tolist() == [2, 1, 2, 1]    # one peak, one late-night
    assert by_zone.loc[74].tolist() == [3, 1, 1, 2]     # Saturday 08:00 is off-peak
    assert 264 not in by_zone.index                       # unknown zones not in zone_day
    assert gh.loc[gh["pu_group"] == "unknown", "trips"].sum() == 1
    assert zd["shared_requests"].sum() == 1
    assert abs(zd["wait_min_sum"].sum() - 4 * 5) < 1e-9  # 4-minute waits, 5 trips in zones 1-263


def test_aggregate_handles_old_schema(raw_yellow_2022):
    zd, gh, audit = aggregate_file(raw_yellow_2022, "yellow", "2022-03")
    assert audit["rows_kept"] == 2
    assert zd["fee_trips"].sum() == 0 and zd["fee_sum"].sum() == 0
    assert zd["driver_pay_sum"].isna().all()
    assert set(gh["pu_group"]) == {"cbd", "airport"}


def test_months():
    assert months("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]
    assert pd.Timestamp(months("2022-01", "2026-08")[-1] + "-01").month == 8
