# How this project was built, phase by phase

Each phase lists the question it answers, what was done, the design decisions and why they were
made, what it produced, and the objections a careful reviewer would raise along with how the
project answers them. Numbers come from `reports/metrics/` and are checked by
`tests/test_reported_numbers.py`.

| Phase | Question | Where |
|---|---|---|
| 0 | What decision does this inform, and what is already known? | this page, README |
| 1 | Can 1.3 billion trip records be reduced to something analysable? | `scripts/build_data.py`, `src/congestion/etl.py` |
| 2 | Can the data be trusted, and which zones are tolled? | notebook 01, `scripts/derive_cbd_zones.py` |
| 3 | What does the raw data show before any model? | notebook 02 |
| 4 | Did the toll reduce trips? (main causal estimate) | notebook 03, `src/congestion/did.py` |
| 5 | What would 2025 have looked like without the toll? | notebook 04, `src/congestion/forecast.py` |
| 6 | Does a synthetic twin of the CBD agree? | notebook 05, `src/congestion/synth.py` |
| 7 | Do the methods agree, and what does it mean for pricing? | notebook 06, decision memo |
| 8 | Can anyone rerun it and get the same numbers, locally and in the cloud? | `tests/`, CI, `scripts/check_metrics.py`, `cloud/` |
| 9 | Can a non-technical reader explore it? | `app/streamlit_app.py` |
| 10 | Review: what was strengthened, and what is left | this page |

---

## Phase 0 · Framing

**Question.** New York's congestion toll started on 5 January 2025. Taxi trips in the zone carry a
$0.75 fee and Uber/Lyft trips $1.50, paid by the rider. Did that reduce taxi and ride-hail trips
in the zone? For an operator, the answer decides whether to offset the fee with promotions, move
supply, or do nothing.

**What was already known.** A study of road speeds (Cook et al., NBER WP 33584) found speeds in the
zone rose 15%. A study of Uber/Lyft trips (Pandey, Guler & Gayah, *Findings*, 2026) used a
2025-vs-2024 difference-in-differences over the first 14 weeks and found trips down about 6% and
shared-ride requests down 38%. A first look in the sister project on driver pay
(`ridehail-driver-pay-optimization`, notebook 04: Uber/Lyft only, 2024 against 2025) found a drop in
trips of 9% or about 1% depending on the design, with a pre-toll drift it could not resolve. That
open question is the starting point here.

**Decisions.**
* *Unit of analysis:* trips by **pickup zone**, because TLC records zones, not addresses, and the
  fee applies to trips that start, end or pass through the zone.
* *Scope:* yellow taxis and Uber/Lyft together, January 2022 to the latest month, so there are two
  pre-toll years and more than a year after it.
* *Three methods with different weaknesses* (difference-in-differences, a forecast, synthetic
  control), because no single observational method is convincing alone.

---

## Phase 1 · Data pipeline

**Question.** How do 168 monthly files (about 30 GB) become something a laptop can analyse?

**What was done.** `scripts/build_data.py` downloads one month at a time, maps the three file
schemas (yellow, green, Uber/Lyft) onto one trip schema with DuckDB, cleans it, and reduces it to
two tables: daily totals per pickup zone (`zone_day`, 1,012,708 rows) and hourly totals per zone
group (`group_hour`). The raw file is then deleted. The run is resumable month by month.

**Decisions.**
* *Store sums, not averages* (fares, minutes, miles, driver pay, waits), so any grouping can be
  re-aggregated exactly.
* *Cleaning rules* are counted per month in an audit table: pickup inside the file's month, 1
  minute to 4 hours, 0–100 miles, fare not negative.
* *Commit the aggregates* (36 MB) so every notebook runs on a fresh clone without the download.

**Result.** 1,296,931,116 rows read, 1,288,695,819 trips kept (99.4%).

**Objection:** "Old files don't have the fee column." The schema mapper fills missing columns with
defaults; `tests/test_etl_zones.py` checks a 2022-style file and a 2025-style file with known
answers.

---

## Phase 2 · Data audit and the toll zone (notebook 01)

**Question.** Can the records be trusted for a before/after comparison, and exactly which taxi zones
are tolled?

**What was done.**
* Kept-row share per service and month (one outlier, November 2025 yellow taxis, traced to refund
  records and a new vendor's zero-length rows, both correctly removed).
* **The zone list was derived from the data** rather than copied: for each zone, the share of
  same-zone trips that paid the fee in January–March 2025. Inside the zone 90–100% pay; outside at
  most 3.5%. That gives 38 tolled zones. Six zones on the 60th Street border pay 14–47% and are left
  out of every comparison (the "donut").
* **Contamination of the comparison group:** trips from the comparison areas that pass through the
  zone also pay. Estimated at 4.8% of rest-of-Manhattan trips and 1.1% of outer-borough trips.
* Green taxis are dropped: they cannot pick up street hails in the zone, and under 1% of the green
  trips recorded there paid.

**Objection:** "Your control group is partly treated." Yes, by about 5%, which biases estimates
slightly toward zero; it is reported, and the within-origin design in Phase 4 avoids it.

---

## Phase 3 · Trends before modelling (notebook 02)

**Question.** What does the raw data say?

**What was done.** Year-on-year growth by area and service, the CBD's share of trips over time, and
the 2024-to-2025 change by hour.

**Finding that shaped the project.** CBD trips fell relative to the rest of the city in 2025, but
**the same gap opened in 2024, a year before the toll**: the CBD's share was already falling about
4% a year against the rest of Manhattan. Any method that ignores this drift will credit the toll
with it.

**Also seen.** The Uber/Lyft decline in 2025 was largest overnight, when the car toll is lowest,
which does not fit a fee that is the same at all hours.

---

## Phase 4 · Difference-in-differences (notebook 03)

**Question.** Did the toll reduce trips starting in the CBD, compared with the rest of Manhattan?

**What was done, in order.**
1. **Quick answer** (2025 vs 2024, zone and week fixed effects, standard errors clustered by zone,
   zones weighted by pre-toll trips): −2.6% (−4.0% to −1.2%).
2. **The same comparison a year earlier, with no toll:** −4.4% (−5.3% to −3.5%). The quick answer
   is mostly the pre-existing trend.
3. **Event study** in quarterly bins: the pre-toll gap slopes down in a straight line, and 2025
   continues that line. Parallel trends fail, so a plain DiD is not credible here.
4. **Main estimate: every zone gets its own linear trend.** +1.0% (95% CI −0.3% to +2.3%).
5. **Fake toll dates.** The same models run on seven pretend start dates in 2023 (pre-toll data
   only). The quick DiD finds a drop every time (−7.1% to −5.2%). The trend-adjusted DiD has no
   such bias but scatters from −5.6% to +7.9% (typical size ±4.3%). This is the honest noise level,
   much wider than the formal interval.
6. **Two-step version:** zone trends fitted on pre-toll weeks only, then extended. This answers the
   objection that trends fitted through 2025 can absorb a toll effect that builds up during the
   year. Result: +1.2%.
7. **Robustness grid:** each service separately, outer boroughs as the comparison, unweighted, from
   2022, and a **within-origin** design (from the same Manhattan zones, trips into the CBD against
   trips elsewhere, which cancels anything that hits a neighbourhood as a whole). All
   trend-adjusted versions sit between −1.3% and +1.2%.
8. **Other outcomes**, each judged against its own fake-date noise: Uber/Lyft trip speed +5.0%,
   taxi speed +4.4%, Uber/Lyft driver pay per hour on trip +2.7% and **shared-ride requests −12.7%**
   all clear the bar. Waits (−5.7%) are suggestive only; trip duration and fare fell, but by about
   as much around a fake toll, so they are not credited. Uber/Lyft trips at peak and overnight hours
   show no clear change.
9. **Price exposure:** the fee is 4.3% of the fare on a CBD trip, for taxis and Uber/Lyft alike.

**Objections and answers.**
* "Zone-specific trends can soak up the effect." Checked with the two-step version (+1.2%) and in
  `tests/test_methods.py`, which plants an effect that grows over time.
* "Your interval is too narrow." Agreed; the fake-date check replaces it as the guide to noise.
* "Shared rides fell everywhere." The estimate is relative to the rest of Manhattan and net of each
  zone's trend, and it clears both fake-date checks.

---

## Phase 5 · Forecast counterfactual (notebook 04)

**Question.** Ignoring other neighbourhoods, what would the CBD's own trips have been in 2025?

**What was done.**
* Three models on daily log trips, trained on July 2022 to December 2024: seasonal naive (same
  weekday a year earlier), a regression (trend, day of week, yearly Fourier terms, holidays,
  Central Park rain, snow and temperature), and **the regression plus LightGBM on its residuals**
  (trees cannot extrapolate a trend, so they only model what the regression misses).
* **Four rolling-origin backtests** that pretend the toll started in October 2023, January, April
  or July 2024. Nothing happened then, so their error is both the accuracy check and the noise
  band. The boosted model wins (weekly error 4.0%).
* 80% and 95% bands from the backtest errors; weekly comparison through August 2026.

**Results.** CBD trips in 2025 were −0.7% against their forecast; the largest no-toll backtest gap
was ±1.4%. The backtests also show the model runs about 1% low for the CBD even with no toll, so
**net of that bias the 2025 gap is +0.3%**. Relative to the rest of Manhattan the CBD did slightly
better than forecast (+1.3% net of bias). In 2026 both areas fall about 4% below forecast, so the
early-2026 drop is Manhattan-wide, not toll-specific.

**Objection:** "The forecast extends the CBD's own trend, so of course it agrees with the
trend-adjusted DiD." Partly true, and stated in the notebook. What it adds is a yardstick based on
the CBD's own seasonality and weather, not on a comparison area.

---

## Phase 6 · Synthetic control (notebook 05)

**Question.** Does a weighted mix of other zones, matched to the CBD before the toll, diverge from it
after?

**What was done.** Non-negative weights summing to one over 186 donor zones, fitted on 2023–2024
weekly log trips; a standard (demeaned) version and a trend-adjusted version; inference by running
the same method on every donor ("placebo in space").

**Results.** The standard twin cannot reproduce the CBD's slide (its pre-toll gap changes sign
between 2023 and 2024), so it repeats the trend: −6.7% (placebo p = 0.18). The trend-adjusted
twin gives +0.3% (placebo p = 0.93): the CBD's 2025 result is ordinary among zones with no toll.

---

## Phase 7 · Synthesis and decision (notebook 06, decision memo)

**Do the methods agree?** Every method that ignores the pre-toll trend finds a drop, and so does the
no-toll placebo year. Every method that accounts for it lands between −0.7% and +2.3%.

**How price-sensitive is demand?** The fee raised the price of a CBD trip by 4.3%. The forecast
check allows trips no lower than −2.1%, which rules out a price elasticity much stronger than
−0.5. The DiD alone is too noisy to say much (its bound is about −1.9).

**Recommendation** (memo): no broad promotions to offset the fee; test covering part of the fee on
shared rides, the one segment that clearly shrank; plan for faster trips and higher driver earnings
per hour in the zone; re-estimate quarterly and before the car toll rises from $9 to $12 in 2028.

**Caveat carried into every summary:** completed trips reflect riders and drivers together. Faster
trips let the same drivers serve more rides, so a small fall in rider demand could be hidden by
better supply.

---

## Phase 8 · Reproducibility

* `pytest`: the ETL on hand-built files with known answers; zone groups (Python and SQL agree); DiD,
  event study, two-step trends, forecast and synthetic control each recovering a planted effect on
  simulated data; the Athena SQL against the DuckDB ETL; every number quoted in the README, memo,
  summary, this page and `cloud/README.md` against the recorded results; and the app.
* CI: lint; tests; `.ipynb` files match their `.py` sources; every notebook re-run on the committed
  aggregates; the re-computed results compared with the committed ones (`scripts/check_metrics.py`,
  relative tolerance 1e-4).
* Notebooks are written as `.py` (percent format) and built into executed `.ipynb` files by
  `scripts/build_notebooks.py --execute`.
* **The pipeline at scale on AWS** ([`cloud/README.md`](../cloud/README.md)). The aggregation is
  also written as Athena SQL over the raw files in S3, with the infrastructure in one
  CloudFormation template. Decisions:
  * one source of truth: the SQL is generated from the constants in `etl.py` and `zones.py`; a test
    translates it to DuckDB (sqlglot) and checks it against the laptop ETL, and the cloud run is
    reconciled with the committed aggregates for every service and month;
  * schema drift found by reading all 168 file footers first (zone ids INT64 then INT32 from
    February 2023, the fee column only from 2025) and absorbed in the table definitions;
  * partition projection instead of a crawler; a per-query scan cap, expiring query results and a
    cost budget as guardrails; three passes over the data kept over a cheaper but
    harder-to-test intermediate table.
  * Result: 9 queries scanned 52.0 GB in 3.8 minutes for $0.24, and 504 of 504 table × service ×
    month checks match the laptop build. The first run matched 501 of 504: a few timestamps carry
    fractions of a second, which DuckDB and Athena count differently (second boundaries vs whole
    seconds elapsed), leaving 3 sums one second apart. Timestamps are now truncated to the second,
    with a test for the case.

---

## Phase 9 · Interactive app

**Question.** Can someone who will never open a notebook see the answer, check it, and poke at it?

**What was built** (`app/streamlit_app.py`, five tabs):
* **The answer:** the result in four numbers (trips, speed, driver pay per hour, shared rides), why a
  quick before/after comparison misleads, every method side by side, the outcomes that clear the
  fake-date bar (a switch shows the ones that do not), and the pricing view (fee share, elasticity
  bound, fees paid).
* **Trends:** trips by area and service, the CBD's share against its extended pre-toll trend, the
  event study, and a map of the 2025 change by pickup zone.
* **Explore a zone:** any busy neighbourhood's trips, speed, fee share and shared rides before and
  after the toll, against its area and the rest of Manhattan. Labelled as plain before/after
  numbers, so a reader does not mistake them for causal estimates.
* **Forecast vs actual:** weekly trips against the no-toll forecast and its backtest bands.
* **Try a fake toll date:** reruns both DiD designs live on any Monday from July 2023 to January 2024.

**Decisions.** The app reads only committed aggregates and the metrics the notebooks wrote, and calls
the same tested functions, so it cannot drift from the analysis. Every chart has a one-line caption
saying what to look at. `tests/test_app_smoke.py` runs it headless, checks the headline against the
recorded metrics, reproduces a notebook fake-date result, and exercises the zone explorer for a
toll-zone, a Manhattan and an outer-borough zone under every service.

**Deployment.** Streamlit Community Cloud from the repository (main file `app/streamlit_app.py`,
Python 3.12, no secrets); steps in the README.

---

## Phase 10 · Review: what was strengthened, and what is left

**Strengthened after the first full draft.**
1. *Noise, measured honestly.* A single fake-2024 placebo suggested ±2%; seven fake dates showed
   about ±4%. The headline changed from "no effect larger than 2–3%" to "no detectable effect".
2. *Trend leakage.* Added the two-step DiD with pre-toll trends only (+1.2%) and a test that plants
   a growing effect.
3. *Forecast noise.* Two backtests became four, and the 2025 gaps are now also shown net of the
   model's no-toll bias.
4. *Clear-effect rule.* An outcome counts only if it is at least twice both its typical fake-date
   effect and its fake-2024 effect. Waits were downgraded to suggestive under this rule.
5. *New outcomes:* shared-ride requests (a clear effect, consistent in direction with the earlier
   study) and trips by time of day (no clear change).
6. *Business translation:* the fee's share of the fare and the implied bound on price elasticity.
7. *Interpretation:* the trips-versus-demand caveat added to the README and memo.
8. *Scale and portability:* the aggregation runs on AWS S3 + Athena from one CloudFormation
   template, reconciled with the local build for every service and month (Phase 8).
9. *Inference (notebook 07).* A wild cluster restricted bootstrap (Webb weights, 9,999 draws,
   interval by test inversion) confirms the zone-clustered intervals: the main estimate's becomes
   −0.3% to +2.3%. A HonestDiD sensitivity analysis on whole-toll-year event-study bins (quarterly bins
   carry a summer pattern in the gap that a smoothness restriction would read as bends) shows a
   drop of 5% or more stays ruled out until the allowed bend reaches 131% of the pre-toll
   slope, while the year-2 decline is on the pre-toll trend (−0.6% (−2.8% to +1.6%)). Both methods are written
   from scratch and tested: the bootstrap against brute-force refits, the worst-case bias against
   the R package's formula, and the interval's coverage by simulation.

**Left for later.**
* *Trips into the CBD by destination* with exact fee status, and the MTA's vehicle-entry counts as
  a supply-side check (they start only in 2025).
* *A second year of data:* re-run when the 2026 data is complete to settle whether the early-2026
  decline is Manhattan-wide.
