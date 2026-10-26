"""Every headline number quoted in the README, memo and summary matches reports/metrics.

The notebooks write the metrics; this test fails if a document drifts from them.
Each check formats a recorded value the way the documents quote it and looks for it.
"""
import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
DOCS = {name: (ROOT / path).read_text(encoding="utf-8")
        for name, path in [("readme", "README.md"), ("memo", "reports/decision_memo.md"),
                           ("summary", "reports/PROJECT_SUMMARY.md"), ("phases", "docs/PHASES.md")]}


def m(name):
    return json.loads((METRICS / f"{name}.json").read_text())


def pct(x, signed=True):
    s = f"{abs(x) * 100:.1f}%"
    if not signed:
        return s
    return ("+" if x >= 0 else "−") + s          # documents use the true minus sign


did, fc, sc, audit, summ = m("did"), m("forecast"), m("synth"), m("data_audit"), m("summary")
other = {(r["service"], r["outcome"]): r for r in csv.DictReader(open(METRICS / "did_other_outcomes.csv"))}


def ci(d):
    return f"{pct(d['ci_low'])} to {pct(d['ci_high'])}"


PHASES_CHECKS = [
    pct(did["quick_2025_vs_2024"]["effect"]), ci(did["quick_2025_vs_2024"]),
    pct(did["placebo_2024_vs_2023"]["effect"]), ci(did["placebo_2024_vs_2023"]),
    pct(did["main_trend_adjusted_year1"]["effect"]), ci(did["main_trend_adjusted_year1"]),
    f"{pct(did['fake_dates']['quick_range'][0])} to {pct(did['fake_dates']['quick_range'][1])}",
    f"{pct(did['fake_dates']['trend_adjusted_range'][0])} to {pct(did['fake_dates']['trend_adjusted_range'][1])}",
    f"±{did['fake_dates']['trend_adjusted_rms'] * 100:.1f}%",
    pct(did["two_step_pre_trend_year1"]["effect"]),
    f"{pct(did['robustness_trend_adjusted_range'][0])} and {pct(did['robustness_trend_adjusted_range'][1])}",
    pct(summ["speed_uber"]), pct(summ["speed_taxi"]), pct(summ["pay_per_hour_uber"]),
    pct(summ["shared_requests_uber"]), pct(summ["wait_uber"]),
    f"{did['fee_share_of_fare_2025'] * 100:.1f}%",
    pct(fc["gap_2025"]["cbd"]), f"±{fc['placebo_noise_max_abs'] * 100:.1f}%",
    pct(fc["gap_2025_net_of_backtest"]["cbd"]), pct(fc["gap_2025_net_of_backtest"]["cbd_minus_manhattan"]),
    f"weekly error {fc['cbd_weekly_mape_chosen'] * 100:.1f}%",
    f"{pct(sc['standard_effect_2025'])} (placebo p = {sc['standard_p_value']:.2f})",
    f"{pct(sc['trend_adjusted_effect_2025'])} (placebo p = {sc['trend_adjusted_p_value']:.2f})",
    f"between {pct(summ['trend_adjusted_range'][0])} and {pct(summ['trend_adjusted_range'][1])}",
    pct(summ["elasticity"]["forecast_lowest_trip_change"]),
    f"−{abs(summ['elasticity']['forecast_elasticity_bound']):.1f}",
    f"−{abs(summ['elasticity']['did_elasticity_bound']):.1f}",
    f"{audit['rows_raw']:,}", f"{audit['trips_kept']:,}", f"{audit['kept_share'] * 100:.1f}%",
    f"{audit['through_trip_share_manhattan'] * 100:.1f}%", f"{audit['through_trip_share_outer'] * 100:.1f}%",
    f"{sc['n_donors']} donor zones",
]

CHECKS = [(("phases",), x) for x in PHASES_CHECKS] + [
    # (documents, expected text)
    (("readme", "memo", "summary"), pct(did["main_trend_adjusted_year1"]["effect"])),
    (("readme", "memo", "summary"), ci(did["main_trend_adjusted_year1"])),
    (("readme", "summary"), pct(did["quick_2025_vs_2024"]["effect"])),
    (("readme", "summary"), ci(did["quick_2025_vs_2024"])),
    (("memo",), f"{pct(did['quick_2025_vs_2024']['effect'], signed=False)} relative to the rest"),
    (("readme", "summary"), pct(did["placebo_2024_vs_2023"]["effect"])),
    (("readme", "summary"), ci(did["placebo_2024_vs_2023"])),
    (("memo",), f"they fell {pct(did['placebo_2024_vs_2023']['effect'], signed=False)}"),
    (("summary",), pct(did["placebo_trend_adjusted_2024"]["effect"])),
    (("readme", "summary"), f"±{did['fake_dates']['trend_adjusted_rms'] * 100:.1f}%"),
    (("memo",), f"±{did['fake_dates']['trend_adjusted_rms'] * 100:.0f}%"),
    (("readme", "summary"), f"{pct(did['fake_dates']['trend_adjusted_range'][0])} to "
                            f"{pct(did['fake_dates']['trend_adjusted_range'][1])}"),
    (("readme", "summary"), f"{pct(did['fake_dates']['quick_range'][1])} to "
                            f"{pct(did['fake_dates']['quick_range'][0])}"),
    (("readme", "memo", "summary"), pct(did["trend_adjusted_year2_2026"]["effect"])),
    (("summary",), ci(did["trend_adjusted_year2_2026"])),
    (("summary",), f"{pct(did['robustness_trend_adjusted_range'][0])} to "
                   f"{pct(did['robustness_trend_adjusted_range'][1])}"),
    (("readme", "memo", "summary"), pct(fc["gap_2025"]["cbd"])),
    (("readme", "memo", "summary"), f"±{fc['placebo_noise_max_abs'] * 100:.1f}%"),
    (("readme", "summary"), pct(fc["gap_2025"]["cbd_minus_manhattan"])),
    (("memo", "summary"), pct(fc["gap_2026"]["cbd_minus_manhattan"])),
    (("summary",), f"{fc['cbd_weekly_mape_chosen'] * 100:.1f}%; {fc['weeks_outside_95_band'] * 100:.1f}% of weeks"),
    (("readme", "summary"), f"{pct(sc['standard_effect_2025'])} (placebo p = {sc['standard_p_value']:.2f})"),
    (("readme", "memo", "summary"), f"{pct(sc['trend_adjusted_effect_2025'])} (placebo p = "
                                    f"{sc['trend_adjusted_p_value']:.2f})"),
    (("readme", "summary"), f"{sc['n_donors']} donor zones"),
    (("readme",), f"between {pct(summ['trend_adjusted_range'][0])} and {pct(summ['trend_adjusted_range'][1])}"),
    (("summary",), f"span {pct(summ['trend_adjusted_range'][0])} to {pct(summ['trend_adjusted_range'][1])}"),
    (("readme", "memo", "summary"), pct(summ["speed_uber"])),
    (("readme", "memo", "summary"), pct(summ["speed_taxi"])),
    (("readme", "memo", "summary"), pct(summ["pay_per_hour_uber"])),
    (("memo", "summary"), f"{pct(summ['wait_uber'])}"),
    (("readme", "memo", "summary"), pct(summ["shared_requests_uber"])),
    (("readme", "memo", "summary"), pct(did["two_step_pre_trend_year1"]["effect"])),
    (("readme", "memo", "summary"), pct(fc["gap_2025_net_of_backtest"]["cbd"])),
    (("readme", "summary"), pct(fc["gap_2025_net_of_backtest"]["cbd_minus_manhattan"])),
    (("readme", "memo", "summary"), pct(did["fee_share_of_fare_2025"], signed=False)),
    (("memo", "summary"), pct(summ["elasticity"]["forecast_lowest_trip_change"])),
    (("readme", "memo", "summary"), f"−{abs(summ['elasticity']['forecast_elasticity_bound']):.1f}"),
    (("summary",), f"−{abs(summ['elasticity']['did_elasticity_bound']):.1f}"),
    (("readme", "memo", "summary"), f"${audit['fees_2025_usd'] / 1e6:.1f}M"),
    (("readme", "summary"), f"{audit['rows_raw']:,}"),
    (("readme", "summary"), f"{audit['trips_kept']:,}"),
    (("readme", "summary"), f"{audit['kept_share'] * 100:.1f}%"),
    (("memo",), f"{audit['trips_kept'] / 1e9:.2f} billion"),
    (("readme", "summary"), f"{audit['n_cbd_zones']} zones"),
    (("summary",), f"{audit['through_trip_share_manhattan'] * 100:.1f}% of rest-of-Manhattan"),
    (("summary",), f"{audit['through_trip_share_outer'] * 100:.1f}% of outer-borough"),
    (("summary",), f"at most {audit['outside_fee_share_max'] * 100:.1f}% outside"),
]


@pytest.mark.parametrize("docs,expected", CHECKS, ids=[f"{'+'.join(d)}:{e}" for d, e in CHECKS])
def test_number_is_quoted(docs, expected):
    missing = [d for d in docs if expected not in DOCS[d]]
    assert not missing, f"{expected!r} not found in {missing}"


def test_only_clear_outcomes_are_claimed():
    """The README and memo headline only outcomes that clear the fake-date bar."""
    for key in [("Uber/Lyft", "Average trip speed"), ("Yellow taxi", "Average trip speed"),
                ("Uber/Lyft", "Driver pay per hour on trip"), ("Uber/Lyft", "Shared-ride requests")]:
        assert other[key]["clear"] == "True", key
    assert other[("Uber/Lyft", "Wait from request to pickup")]["clear"] == "False"
    assert other[("Taxi + Uber/Lyft", "Trips (main estimate)")]["clear"] == "False"
