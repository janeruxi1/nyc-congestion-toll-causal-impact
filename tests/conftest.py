import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def raw_hvfhv(tmp_path):
    """A tiny Uber/Lyft-style monthly file with known answers."""
    t = pd.Timestamp
    rows = [
        # pickup,              mins, PU,  DO,  fare, pay, fee, shared
        (t("2025-01-06 08:00"), 10, 161, 237, 20.0, 15.0, 1.5, "N"),   # CBD -> border, peak (Mon)
        (t("2025-01-06 23:00"), 12, 161, 161, 10.0, 8.0, 1.5, "Y"),    # CBD -> CBD, off-peak
        (t("2025-01-06 08:30"), 20, 74, 161, 30.0, 22.0, 1.5, "N"),    # Harlem -> CBD
        (t("2025-01-06 09:00"), 15, 74, 75, 12.0, 9.0, 0.0, "N"),      # Harlem -> Harlem
        (t("2025-01-11 08:00"), 15, 74, 75, 12.0, 9.0, 0.0, "N"),      # Saturday 08:00: off-peak
        (t("2025-01-06 10:00"), 0.5, 74, 75, 12.0, 9.0, 0.0, "N"),     # 30 s: dropped
        (t("2024-12-31 10:00"), 15, 74, 75, 12.0, 9.0, 0.0, "N"),      # wrong month: dropped
        (t("2025-01-07 10:00"), 15, 264, 75, 12.0, 9.0, 0.0, "N"),     # unknown zone
    ]
    df = pd.DataFrame(rows, columns=["pickup_datetime", "mins", "PULocationID", "DOLocationID",
                                     "base_passenger_fare", "driver_pay", "cbd_congestion_fee",
                                     "shared_request_flag"])
    df["dropoff_datetime"] = df["pickup_datetime"] + pd.to_timedelta(df["mins"], unit="m")
    df["request_datetime"] = df["pickup_datetime"] - pd.Timedelta(minutes=4)
    df["trip_miles"] = 3.0
    df["tips"] = 1.0
    df["shared_match_flag"] = df["shared_request_flag"]
    df = df.drop(columns="mins")
    path = tmp_path / "fhvhv_tripdata_2025-01.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def raw_yellow_2022(tmp_path):
    """A pre-toll yellow file: no cbd_congestion_fee column, 'Airport_fee' casing."""
    t = pd.Timestamp
    df = pd.DataFrame({
        "tpep_pickup_datetime": [t("2022-03-01 12:00"), t("2022-03-02 12:00")],
        "tpep_dropoff_datetime": [t("2022-03-01 12:10"), t("2022-03-02 12:30")],
        "PULocationID": [230, 132],
        "DOLocationID": [161, 230],
        "trip_distance": [1.2, 17.0],
        "fare_amount": [8.0, 52.0],
        "tip_amount": [2.0, 10.0],
        "Airport_fee": [0.0, 1.25],
    })
    path = tmp_path / "yellow_tripdata_2022-03.parquet"
    df.to_parquet(path, index=False)
    return path


def make_panel(effect=0.10, n_treated=20, n_control=60, weeks=150, post_from=100, seed=0,
               treated_trend_per_year=0.0):
    """Zone-week panel with zone and week effects and a known treatment effect."""
    rng = np.random.default_rng(seed)
    wk = pd.date_range("2023-01-02", periods=weeks, freq="7D")
    rows = []
    week_fx = rng.normal(0, 0.05, weeks).cumsum()
    for z in range(n_treated + n_control):
        treated = z < n_treated
        a = rng.normal(6, 1)
        for i, w in enumerate(wk):
            post = i >= post_from
            drift = treated_trend_per_year * i / 52.18 if treated else 0.0
            y = (a + week_fx[i] + drift + (np.log1p(effect) if treated and post else 0)
                 + rng.normal(0, 0.05))
            rows.append({"pu_zone": z, "week": w, "treated": treated, "post": post, "log_trips": y,
                         "pre_mean_trips": float(np.exp(a))})
    return pd.DataFrame(rows)
