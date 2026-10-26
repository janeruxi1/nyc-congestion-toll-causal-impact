"""Analysis panels built from the committed aggregates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .zones import AIRPORT_ZONES, BORDER_ZONES, CBD_ZONES, zone_group

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REFERENCE = ROOT / "data" / "reference"

POLICY_START = pd.Timestamp("2025-01-05")      # first tolled day (a Sunday)
# Weekly panels use Monday-start weeks. The week of 30 Dec 2024 holds one
# tolled day, so it is dropped; the first post week starts 6 Jan 2025.
FIRST_POST_WEEK = pd.Timestamp("2025-01-06")
DROP_WEEKS = (pd.Timestamp("2024-12-30"),)
FIRST_FULL_WEEK = pd.Timestamp("2022-01-03")   # 1-2 Jan 2022 belong to a partial week

# Analysis windows (Monday-start weeks). Two pre-policy years, then each
# post-policy year. 2022 is left out of the main window: trips were still
# rebounding from the pandemic, and the CBD rebounded faster.
WINDOW_START = pd.Timestamp("2023-01-09")
YEAR1_END = pd.Timestamp("2025-12-28")      # last week starting in 2025 that ends in 2025

# Green taxis cannot pick up street hails in the CBD (see notebook 01), so the
# analysis covers yellow taxis and Uber/Lyft.
ANALYSIS_SERVICES = ("yellow", "hvfhv")

SERVICE_LABEL = {"yellow": "Yellow taxi", "green": "Green taxi", "hvfhv": "Uber/Lyft"}


def load_zone_day(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or PROCESSED / "zone_day.parquet")
    df["service"] = df["service"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["group"] = df["pu_zone"].map(zone_group)
    return df


def load_group_hour(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or PROCESSED / "group_hour.parquet")
    for c in ("service", "pu_group"):
        df[c] = df[c].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_weather(path: Path | None = None) -> pd.DataFrame:
    w = pd.read_csv(path or REFERENCE / "weather_central_park.csv", parse_dates=["date"])
    return w


def week_start(dates: pd.Series) -> pd.Series:
    return dates - pd.to_timedelta(dates.dt.day_of_week, unit="D")


def complete_dates(df: pd.DataFrame, last_date: pd.Timestamp | None = None) -> pd.Timestamp:
    """Last date with data for every service (the post period ends here)."""
    ends = df.groupby("service")["date"].max()
    end = ends.min()
    return min(end, last_date) if last_date is not None else end


def zone_week_panel(zone_day: pd.DataFrame, services: tuple[str, ...] = ANALYSIS_SERVICES,
                    min_pre_trips_per_week: float = 50.0) -> pd.DataFrame:
    """Zone x week panel for the origin-zone comparison (design A).

    Outcome for CBD zones: every trip that starts there (all of them pay).
    Outcome for comparison zones: trips that start there and do *not* end in
    the CBD, so tolled trips into the zone are not counted as untreated.
    Border zones and airports are dropped; zones with fewer than
    `min_pre_trips_per_week` trips in an average pre-policy week are dropped
    (too noisy to log).
    """
    df = zone_day[zone_day["service"].isin(services)].copy()
    df = df[~df["pu_zone"].isin(BORDER_ZONES | AIRPORT_ZONES)]
    df = df[df["group"].isin(["cbd", "manhattan_north", "outer"])]
    df["treated"] = df["pu_zone"].isin(CBD_ZONES)
    df["outcome_trips"] = np.where(df["treated"], df["trips"], df["trips"] - df["trips_to_cbd"])
    df["week"] = week_start(df["date"])
    df = df[~df["week"].isin(DROP_WEEKS) & (df["week"] >= FIRST_FULL_WEEK)]
    end = complete_dates(zone_day)
    last_full_week = week_start(pd.Series([end + pd.Timedelta(days=1)]))[0] - pd.Timedelta(days=7)
    df = df[df["week"] <= last_full_week]

    agg = {c: "sum" for c in ["outcome_trips", "trips", "trips_to_cbd", "fee_trips",
                              "fare_sum", "driver_pay_sum", "miles_sum", "minutes_sum",
                              "wait_trips", "wait_min_sum", "shared_requests", "shared_matches"]}
    p = df.groupby(["pu_zone", "week", "treated"], as_index=False).agg(agg)
    # Balance: every zone x week (zero trips where a zone had none)
    zones, weeks = p["pu_zone"].unique(), np.sort(p["week"].unique())
    full = pd.MultiIndex.from_product([zones, weeks], names=["pu_zone", "week"]).to_frame(index=False)
    p = full.merge(p.drop(columns="treated"), on=["pu_zone", "week"], how="left").fillna(0)
    p["treated"] = p["pu_zone"].isin(CBD_ZONES)
    p["post"] = p["week"] >= FIRST_POST_WEEK
    pre_mean = p[~p["post"]].groupby("pu_zone")["outcome_trips"].mean()
    keep = pre_mean[pre_mean >= min_pre_trips_per_week].index
    p = p[p["pu_zone"].isin(keep)].copy()
    p["pre_mean_trips"] = p["pu_zone"].map(pre_mean)
    p["log_trips"] = np.log(p["outcome_trips"] + 1)
    return p.sort_values(["pu_zone", "week"]).reset_index(drop=True)


def flow_week_panel(zone_day: pd.DataFrame, services: tuple[str, ...] = ANALYSIS_SERVICES,
                    min_pre_trips_per_week: float = 20.0) -> pd.DataFrame:
    """Origin x destination-type x week panel (design B).

    From each comparison zone (rest of Manhattan and outer boroughs), trips
    *into* the CBD (tolled) are compared with trips *elsewhere* (not tolled
    unless they pass through). Anything that hits a neighbourhood as a whole,
    such as local events, weather or a new subway line, cancels out.
    """
    df = zone_day[zone_day["service"].isin(services)].copy()
    df = df[df["group"].isin(["manhattan_north", "outer"])]
    df["week"] = week_start(df["date"])
    df = df[~df["week"].isin(DROP_WEEKS) & (df["week"] >= FIRST_FULL_WEEK)]
    end = complete_dates(zone_day)
    last_full_week = week_start(pd.Series([end + pd.Timedelta(days=1)]))[0] - pd.Timedelta(days=7)
    df = df[df["week"] <= last_full_week]
    w = df.groupby(["pu_zone", "week"], as_index=False)[["trips", "trips_to_cbd"]].sum()
    to_cbd = w.assign(flow="to_cbd", flow_trips=w["trips_to_cbd"])
    other = w.assign(flow="elsewhere", flow_trips=w["trips"] - w["trips_to_cbd"])
    p = pd.concat([to_cbd, other])[["pu_zone", "week", "flow", "flow_trips"]]
    p["treated"] = p["flow"].eq("to_cbd")
    p["post"] = p["week"] >= FIRST_POST_WEEK
    pre = p[~p["post"]].groupby(["pu_zone", "flow"])["flow_trips"].mean().unstack()
    keep = pre[(pre.min(axis=1) >= min_pre_trips_per_week)].index
    p = p[p["pu_zone"].isin(keep)].copy()
    p["unit"] = p["pu_zone"].astype(str) + "_" + p["flow"]
    p["pre_mean_trips"] = p.set_index(["pu_zone", "flow"]).index.map(pre.stack())
    p["log_trips"] = np.log(p["flow_trips"] + 1)
    return p.sort_values(["unit", "week"]).reset_index(drop=True)


def daily_series(zone_day: pd.DataFrame, services: tuple[str, ...] = ANALYSIS_SERVICES
                 ) -> pd.DataFrame:
    """Daily trips: CBD-origin trips, and trips from the rest of Manhattan and
    the outer boroughs that do not end in the CBD (border zones and airports
    excluded)."""
    df = zone_day[zone_day["service"].isin(services)
                  & zone_day["group"].isin(["cbd", "manhattan_north", "outer"])].copy()
    df["outcome_trips"] = np.where(df["group"] == "cbd", df["trips"], df["trips"] - df["trips_to_cbd"])
    out = df.pivot_table(index="date", columns="group", values="outcome_trips", aggfunc="sum").fillna(0)
    out = out.rename(columns={"manhattan_north": "manhattan"})[["cbd", "manhattan", "outer"]]
    end = complete_dates(zone_day)
    out = out[out.index <= end]
    out.columns.name = None
    return out.reset_index()


def window(panel: pd.DataFrame, start: pd.Timestamp | str, end: pd.Timestamp | str,
           post_from: pd.Timestamp | str = FIRST_POST_WEEK) -> pd.DataFrame:
    """Rows with `start <= week <= end`; `post` marks weeks from `post_from`."""
    out = panel[(panel["week"] >= pd.Timestamp(start)) & (panel["week"] <= pd.Timestamp(end))].copy()
    out["post"] = out["week"] >= pd.Timestamp(post_from)
    return out


def zone_week_outcomes(zone_day: pd.DataFrame, service: str, min_trips: int = 50) -> pd.DataFrame:
    """Per-trip outcomes by origin zone and week for one service.

    Speed, duration, waits and pay are averaged over *all* trips from a zone
    (the aggregates do not split these sums by destination), so comparison
    zones include their trips into the CBD; notebook 01 sizes that overlap.
    """
    df = zone_day[(zone_day["service"] == service)
                  & zone_day["group"].isin(["cbd", "manhattan_north", "outer"])].copy()
    df["week"] = week_start(df["date"])
    df = df[~df["week"].isin(DROP_WEEKS) & (df["week"] >= FIRST_FULL_WEEK)]
    cols = ["trips", "trips_peak", "miles_sum", "minutes_sum", "fare_sum", "driver_pay_sum",
            "wait_trips", "wait_min_sum", "fee_sum", "shared_requests"]
    w = df.groupby(["pu_zone", "week", "group"], as_index=False)[cols].sum(min_count=1)
    w = w[w["trips"] >= min_trips].copy()
    w["treated"] = w["group"].eq("cbd")
    w["post"] = w["week"] >= FIRST_POST_WEEK
    w["log_mph"] = np.log(w["miles_sum"] / (w["minutes_sum"] / 60))
    w["log_minutes"] = np.log(w["minutes_sum"] / w["trips"])
    w["log_miles"] = np.log(w["miles_sum"] / w["trips"])
    w["log_fare"] = np.log(w["fare_sum"] / w["trips"])
    w["log_trips_peak"] = np.log1p(w["trips_peak"])
    w["log_trips_offpeak"] = np.log1p(w["trips"] - w["trips_peak"])
    if service == "hvfhv":
        w["log_pay_per_trip"] = np.log(w["driver_pay_sum"] / w["trips"])
        w["log_pay_per_hour"] = np.log(w["driver_pay_sum"] / (w["minutes_sum"] / 60))
        w["log_wait"] = np.log(w["wait_min_sum"] / w["wait_trips"])
        # Shared-ride requests (UberX Share / Lyft Shared): there is no toll
        # discount for sharing, so a $1.50 fee per trip weighs more on a cheap fare.
        w["log_shared_requests"] = np.log1p(w["shared_requests"])
    return w.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def taxi_share_panel(zone_day: pd.DataFrame, min_trips: int = 100) -> pd.DataFrame:
    """log(yellow trips / Uber-Lyft trips) by origin zone and week.

    Taxis pay $0.75 per CBD trip and Uber/Lyft $1.50, so if riders respond to
    the fee, taxis should gain share inside the zone relative to elsewhere.
    """
    df = zone_day[zone_day["service"].isin(["yellow", "hvfhv"])
                  & zone_day["group"].isin(["cbd", "manhattan_north"])].copy()
    df["week"] = week_start(df["date"])
    df = df[~df["week"].isin(DROP_WEEKS) & (df["week"] >= FIRST_FULL_WEEK)]
    w = df.pivot_table(index=["pu_zone", "week", "group"], columns="service",
                       values="trips", aggfunc="sum").reset_index()
    w = w[(w["yellow"] >= min_trips) & (w["hvfhv"] >= min_trips)].copy()
    w["treated"] = w["group"].eq("cbd")
    w["post"] = w["week"] >= FIRST_POST_WEEK
    w["log_taxi_ratio"] = np.log(w["yellow"] / w["hvfhv"])
    w["weight"] = w["yellow"] + w["hvfhv"]
    return w.reset_index(drop=True)
