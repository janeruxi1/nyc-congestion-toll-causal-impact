# Project summary: NYC congestion toll and taxi / Uber/Lyft trips

One page: what was built, what was found, and where each number comes from. Every number below is
written by a notebook to `reports/metrics/` and checked against those files by
`tests/test_reported_numbers.py`.

## 60-second overview

New York's congestion toll started on 5 January 2025 for vehicles entering Manhattan below 60th
Street (the CBD). Taxi trips there carry a $0.75 fee and Uber/Lyft trips $1.50. Using all
1,288,695,819 cleaned TLC trip records from January 2022 to August 2026, the project estimates the
toll's effect on trips starting in the CBD with difference-in-differences, a forecast
counterfactual and synthetic control.

**Headline.** A quick 2025-vs-2024 comparison shows −2.6%, but the same comparison in a no-toll year
shows −4.4%: the CBD was already losing share. Accounting for that trend, the year-one effect is
**+1.0% (95% CI −0.3% to +2.3%)**, and the three trend-aware methods span −0.7% to +2.3%. Trips got
faster (+5.0% Uber/Lyft, +4.4% taxi), Uber/Lyft driver pay per hour on a trip rose (+2.7%) and
shared-ride requests fell (−12.7%). No method can rule out a small effect on trips: on seven fake
toll dates the main method finds "effects" of typically ±4.3%. The fee added 4.3% to the price of a
CBD trip, and the forecast check rules out a price elasticity much stronger than −0.5.

## Components

| Part | What it does | Source |
|---|---|---|
| ETL | 168 monthly files, 1,296,931,116 rows, reduced with DuckDB to daily totals per pickup zone; 99.4% of rows kept | `scripts/build_data.py`, `src/congestion/etl.py`, notebook 01 |
| CBD zones | 38 zones derived from the fee: same-zone trips pay 90–100% of the time inside, at most 3.5% outside; 6 border zones excluded | `scripts/derive_cbd_zones.py`, notebook 01 |
| Contamination check | 4.8% of rest-of-Manhattan and 1.1% of outer-borough comparison trips pass through the zone and pay | notebook 01 |
| Trends | CBD share falling about 4.1% a year against the rest of Manhattan (6.1% against the outer boroughs) in 2023–2024 | notebook 02 |
| Difference-in-differences | Zone and week fixed effects, zone-clustered SEs, trips-weighted; placebo year; event study; zone-specific trends (whole window, and a two-step version with pre-toll trends only); seven fake toll dates | `src/congestion/did.py`, notebook 03 |
| Forecast | Seasonal naive, regression with weather and holidays, and LightGBM on the regression's residuals; four rolling-origin backtests with no toll; gaps net of backtest bias | `src/congestion/forecast.py`, notebook 04 |
| Synthetic control | 186 donor zones, non-negative weights, standard and trend-adjusted, placebo in space | `src/congestion/synth.py`, notebook 05 |
| Summary | Methods side by side, price-elasticity bounds, hero figure | notebook 06 |
| Robust inference | Wild cluster restricted bootstrap; HonestDiD smoothness sensitivity with fixed-length intervals and break-down points | `src/congestion/inference.py`, notebook 07 |
| AWS run | The same aggregation as Athena SQL over the raw files in S3 (CloudFormation stack); 9 queries, 52.0 GB scanned for $0.24; 504 of 504 table × service × month checks match the local build | `cloud/`, `scripts/cloud_run.py`, `reports/cloud/` |
| Phases | How the project was built, step by step, with the design choices and objections | `docs/PHASES.md` |

## Headline numbers

| Estimate | Value | Notebook |
|---|---|---|
| DiD 2025 vs 2024 (quick) | −2.6% (−4.0% to −1.2%) | 03 |
| DiD 2024 vs 2023 (no toll, placebo) | −4.4% (−5.3% to −3.5%) | 03 |
| **DiD with zone trends, year 1 (main)** | **+1.0% (−0.3% to +2.3%)** | 03 |
| Two-step: zone trends from pre-toll weeks only | +1.2% | 03 |
| Same method on a fake January 2024 toll | +2.0% | 03 |
| Seven fake toll dates (Jul 2023 – Jan 2024): quick DiD | −5.2% to −7.1% | 03 |
| Seven fake toll dates: trend-adjusted DiD | −5.6% to +7.9% (typical ±4.3%) | 03 |
| Robustness grid (services, comparison groups, weights, within-origin) | −1.3% to +1.2% | 03 |
| Year 2 (Jan–Aug 2026), zone trends | −3.6% (−6.0% to −1.1%) | 03 |
| Forecast: CBD vs its own forecast, 2025 | −0.7% | 04 |
| Forecast: largest gap in no-toll backtests | ±1.4% | 04 |
| Forecast: CBD gap minus Manhattan gap, 2025 / 2026 | +2.3% / +0.2% | 04 |
| Forecast: 2025 gaps net of the average no-toll backtest gap (CBD / relative) | +0.3% / +1.3% | 04 |
| Forecast model accuracy (CBD, weekly MAPE, four backtests) | 4.0%; 2.3% of weeks outside the 95% band | 04 |
| Synthetic control, standard | −6.7% (placebo p = 0.18) | 05 |
| Synthetic control, trend-adjusted | +0.3% (placebo p = 0.93) | 05 |
| Main estimate, wild cluster bootstrap interval | −0.3% to +2.3% | 07 |
| HonestDiD, year 1, pre-toll trend continues exactly (M = 0) | +1.5% (+0.3% to +2.7%) | 07 |
| HonestDiD: bend needed before a −5% drop fits the interval | 131% of the pre-toll slope | 07 |
| HonestDiD, year 2 (Jan–Aug 2026), M = 0 | −0.6% (−2.8% to +1.6%) | 07 |

## Other outcomes (trend-adjusted DiD, year 1, each at least twice its fake-date effects)

| Outcome | Change | Notebook |
|---|---|---|
| Uber/Lyft average trip speed | +5.0% | 03 |
| Yellow taxi average trip speed | +4.4% | 03 |
| Uber/Lyft driver pay per hour on trip | +2.7% | 03 |
| Uber/Lyft shared-ride requests | −12.7% | 03 |
| CBD fees paid by taxi and ride-hail riders, 2025 | $148.7M | 01 |

Uber/Lyft waits changed by −5.7%, but move almost as much on fake dates (suggestive only). Trip duration and
fare per trip also fell, but by about as much around a fake January 2024 toll, so they are not
attributed to the toll. Uber/Lyft trips at car-toll peak hours and overnight show no clear change.

## Price sensitivity (notebook 06)

| Quantity | Value |
|---|---|
| Fee as a share of the fare, CBD trips, 2025 | 4.3% |
| Lowest trip change the forecast check allows (−0.7% − 1.4%) | −2.1% |
| Implied bound on the price elasticity | about −0.5 |
| Same bound from the DiD alone (effect − 2 × fake-date noise) | about −1.9 |

## Reading order

1. `README.md` → this page → `reports/decision_memo.md`
2. Notebook 02 (the trend that a quick comparison misses), then 03 (main estimate)
3. Notebooks 04 and 05 (independent checks), 06 (side by side), 07 (how far the answer can move)
4. Notebook 01 and `data/README.md` for the data and the zone list

## Engineering

* `pytest`: ETL on a hand-built file with known answers, zone groups, the Athena SQL against the
  DuckDB ETL (translated with sqlglot), DiD / event study / forecast / synthetic control recovering
  known effects on simulated panels, the reported numbers, and an app smoke test.
* CI: lint, tests, every notebook re-run on the committed aggregates, `.ipynb` and Athena SQL files
  checked against their sources, and re-computed metrics compared with the committed ones.
* AWS: the recorded Athena run reproduces every committed aggregate (`reports/cloud/`); the first
  run found, and the SQL now handles, a one-second difference in how DuckDB and Athena count time.
