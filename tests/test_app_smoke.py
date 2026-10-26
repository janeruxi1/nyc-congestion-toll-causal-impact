"""The Streamlit app runs end to end on the committed aggregates and shows the recorded results."""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    return at


def test_headline_matches_recorded_metrics(app):
    did = json.loads((ROOT / "reports" / "metrics" / "did.json").read_text())
    values = {m.label: m.value for m in app.metric}
    assert values["Trips from the toll zone"] == f"{did['main_trend_adjusted_year1']['effect']:+.1%}"
    assert values["2025 vs 2024"] == f"{did['quick_2025_vs_2024']['effect']:+.1%}"


def test_widgets_rerun_without_errors(app):
    app.radio(key="svc").set_value("Yellow taxi").run()
    app.radio(key="fc_series").set_value("Rest of Manhattan").run()
    app.toggle(key="show_all_outcomes").set_value(True).run()
    assert not app.exception, app.exception


def test_zone_explorer_handles_every_kind_of_zone(app):
    """A toll-zone zone, a Manhattan zone and an outer-borough zone, for each service."""
    labels = app.selectbox(key="zone_pick").options
    picks = [next(lb for lb in labels if "toll zone" in lb),
             next(lb for lb in labels if "Manhattan" in lb and "toll zone" not in lb),
             next(lb for lb in labels if "Brooklyn" in lb)]
    for lb in picks:
        app.selectbox(key="zone_pick").set_value(lb).run()
        for svc in ["Taxi + Uber/Lyft", "Uber/Lyft", "Yellow taxi"]:
            app.radio(key="zone_svc").set_value(svc).run()
            assert not app.exception, (lb, svc, app.exception)


def test_fake_toll_date_reproduces_notebook(app):
    """The app's fake-date DiD matches notebook 03's run for the same date."""
    import csv
    rows = {r["fake_start"]: r for r in csv.DictReader(open(ROOT / "reports" / "metrics" / "did_fake_dates.csv"))}
    app.select_slider(key="fake_start").set_value("2023-10-23").run()
    assert not app.exception, app.exception
    values = {m.label: m.value for m in app.metric}
    assert values["Trend-adjusted DiD"] == f"{float(rows['2023-10-23']['trend_adjusted']):+.1%}"
    assert values["Quick DiD (no trends)"] == f"{float(rows['2023-10-23']['quick']):+.1%}"
