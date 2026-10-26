# Decision memo: responding to the Manhattan congestion toll

**To:** Head of Marketplace (taxi and ride-hail operations) · **From:** Xi Ru, Data Science · **Status:** recommendation
**Based on public NYC TLC trip records, January 2022 – August 2026 (1.29 billion trips). Observational study; see Limitations.**

## Recommendation

1. **Do not fund broad promotions to offset the CBD fee.** Riders absorbed the $1.50 (Uber/Lyft) and
   $0.75 (taxi) per-trip fee, about 4.3% of the fare, without a detectable drop in trips during the
   toll's first year.
2. **Test covering part of the fee on shared rides.** Shared-ride requests from the CBD fell by about
   an eighth (−12.7%). The fee has no discount for sharing, so it weighs most on the cheapest rides.
   A small test (for example, half the fee refunded on shared rides in a few zones) would show
   whether the requests come back.
3. **Plan for faster, not fewer, trips in the zone.** Trips starting in the CBD are about 5% faster,
   and Uber/Lyft drivers earn about 3% more per hour on a CBD trip. Reflect this in supply
   positioning and in pay models that use time-in-zone.
4. **Re-estimate every quarter**, starting with the early-2026 decline (below), and again before the
   car toll rises from $9 to $12 in 2028.

## Why

1. **The quick comparison blames the toll for a trend that started earlier.** Trips from the CBD fell
   2.6% relative to the rest of Manhattan from 2024 to 2025 (95% CI −4.0% to −1.2%). But from 2023 to
   2024, with no toll, they fell 4.4%. The CBD has been losing share of trips by about 4% a year since
   2023.
2. **Three methods that account for that trend agree on roughly zero.**

| Method | Effect on CBD trips, 2025 |
|---|---|
| Difference-in-differences with zone trends (main) | **+1.0%** (95% CI −0.3% to +2.3%) |
| CBD against its own no-toll forecast | −0.7% (no-toll backtests: ±1.4%) |
| Synthetic control, trend-adjusted | +0.3% (placebo p = 0.93) |

   Fitting each zone's trend on pre-toll weeks only (so the trend cannot absorb a toll effect that
   builds up during 2025) gives +1.2%, and the forecast check, net of its small bias in no-toll
   backtests, gives +0.3%.

   None of these can rule out a small effect. Run on seven fake toll dates in 2023, the main
   method finds "effects" of typically ±4%; the forecast misses by up to ±1.4% in no-toll
   backtests. The honest reading is **no detectable effect**, with three methods that fail in
   different ways all landing near zero.
3. **The toll changed how trips run and which rides people take, not how many trips there are.** Measured the same way, each at least
   twice the size of what the method finds on fake toll dates:

| Outcome (trips starting in the CBD) | Change |
|---|---|
| Uber/Lyft average trip speed | +5.0% |
| Yellow taxi average trip speed | +4.4% |
| Uber/Lyft driver pay per hour on trip | +2.7% |
| Uber/Lyft shared-ride requests | −12.7% |

   Uber/Lyft waits also fell (−5.7%), but that outcome moves almost as much on fake dates, so it
   is suggestive only.
4. **Demand in the zone is not very price-sensitive.** The fee raised the price of a CBD trip by
   4.3%. The forecast check, the tightest of the methods, puts trips no lower than −2.1%, which rules
   out a price elasticity much stronger than −0.5. Riders paid $148.7M in per-trip CBD fees in 2025.

## What to watch

* **Early 2026.** Against zone trends, CBD trips changed by −3.6% in January–August 2026. The forecast
  check shows the rest of Manhattan fell short of its forecast by about the same amount (CBD minus
  Manhattan: +0.2%), so this looks Manhattan-wide rather than toll-specific. If the gap persists
  into a second quarter, revisit recommendation 1.
* **Hour of day.** The 2025 decline in Uber/Lyft trips from the CBD was largest overnight, when the
  car toll is lowest; this points away from the fee as its cause.

## Limitations

* The data shows completed trips, which reflect riders and drivers together. Faster trips let the
  same drivers serve more rides, so a small fall in rider demand could be masked by better supply.
* No randomized control group: the main estimate assumes each zone would have continued its
  2023–2024 trend.
* About 5% of comparison trips pass through the zone and pay the fee, which biases estimates slightly
  toward zero.
* Pickup zones, not exact locations; trips are assigned to where they start.
* Other changes in 2025–26 (return-to-office, fare and driver-pay rules) affect Manhattan broadly; the
  comparison groups absorb what is common, not what is specific to the CBD.
* Legal status: a federal court ruled in March 2026 that the federal attempt to end the program was
  unlawful; an appeal remains possible.

Full analysis: notebooks 01–06 in this repository.
