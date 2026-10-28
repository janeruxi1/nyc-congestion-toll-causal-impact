# Data

All data here is public. The raw trip records are not committed (about 30 GB); the aggregates
built from them are, so every notebook and the app run on a fresh clone.

## Sources

| File | Source | Notes |
|---|---|---|
| TLC trip records (not committed) | [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), monthly parquet files | Yellow taxi, green taxi and high-volume for-hire vehicle (Uber, Lyft) trips, January 2022 – August 2026 (168 files). `cbd_congestion_fee` is present from January 2025. |
| `reference/taxi_zone_lookup.csv` | TLC taxi zone lookup | 265 location ids with borough and zone name (264/265 are unknown / outside NYC) |
| `reference/taxi_zones.geojson` | TLC taxi zone shapes, converted to GeoJSON (WGS84) | Used for maps in the app |
| `reference/weather_central_park.csv` | NOAA GHCN-Daily, Central Park station (USW00094728) | Daily rain and snow (inches), max and min temperature (°F), 2022-01-01 onward |
| `reference/cbd_zone_evidence.csv` | Written by `scripts/derive_cbd_zones.py` | For January–March 2025, the share of same-zone trips that paid the CBD fee, per zone; evidence for the zone list in `src/congestion/zones.py` |

## Processed aggregates (`processed/`)

Built by `python scripts/build_data.py --first 2022-01 --last 2026-08`. Each monthly file is
cleaned and reduced with DuckDB, then deleted. Cleaning keeps trips whose pickup falls inside the
file's month, lasting 1 minute to 4 hours, with distance 0–100 miles and a non-negative fare.

**`zone_day.parquet`**: one row per service × date × pickup zone (1,012,708 rows).

| Column | Meaning |
|---|---|
| `service` | `yellow`, `green` or `hvfhv` (Uber/Lyft) |
| `date`, `pu_zone` | Pickup date and TLC pickup zone id |
| `trips`, `trips_peak` | Trips, and trips during car-toll peak hours (weekdays 5am–9pm, weekends 9am–9pm) |
| `trips_to_cbd` | Trips whose drop-off zone is inside the CBD |
| `fee_trips`, `fee_sum` | Trips with a CBD fee, and the fees in dollars |
| `fare_sum`, `tips_sum` | Fare before fees and tips (`base_passenger_fare` for Uber/Lyft, `fare_amount` for taxis), and tips |
| `driver_pay_sum` | Driver pay (Uber/Lyft only) |
| `miles_sum`, `minutes_sum` | Trip distance and duration |
| `wait_trips`, `wait_min_sum` | Uber/Lyft trips with a valid request time, and minutes from request to pickup |
| `shared_requests`, `shared_matches` | Uber/Lyft shared-ride requests and matches |

**`group_hour.parquet`**: one row per service × date × hour × pickup group (592,253 rows), with
the same sums. Groups: `cbd`, `border` (the six 60th Street zones), `manhattan_north`, `outer`,
`airport`, `unknown`.

**`audit.csv`**: per service and month, raw rows, rows inside the month, rows kept and fee-paying
trips.

Sums are kept rather than averages so that any grouping can be re-aggregated exactly.

The same three tables can also be built in the cloud, with Amazon Athena over the raw files in S3
(`scripts/cloud_run.py`, guide in [`cloud/README.md`](../cloud/README.md)). The recorded run matches
these files for every service and month (`reports/cloud/reconciliation.csv`).
