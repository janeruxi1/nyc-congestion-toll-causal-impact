"""NYC congestion toll and taxi / Uber-Lyft trips -- interactive companion app.

Tabs:
    1. The answer            -- the headline, the methods side by side, what did change
    2. Trends                -- trips by area, the CBD's share, the event study, a zone map
    3. Explore a zone        -- one neighbourhood's trips, speed and fees before and after
    4. Forecast vs actual    -- the no-toll forecast with its backtest bands
    5. Try a fake toll date  -- the same DiD on a date when nothing happened

Reads only the committed aggregates in data/processed and the results the
notebooks wrote to reports/metrics, and calls the tested code in src/congestion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from congestion.did import fit_did  # noqa: E402
from congestion.panel import (  # noqa: E402
    load_zone_day, week_start, window, zone_week_panel,
)
from congestion.zones import CBD_ZONES, zone_group  # noqa: E402

METRICS = ROOT / "reports" / "metrics"
REPO_URL = "https://github.com/janeruxi1/nyc-congestion-toll-causal-impact"
TOLL_DAY = pd.Timestamp("2025-01-05")
COLORS = {"cbd": "#C2410C", "manhattan_north": "#1D4ED8", "outer": "#64748B"}
GROUP_LABEL = {"cbd": "Toll zone (CBD)", "manhattan_north": "Rest of Manhattan", "outer": "Outer boroughs"}
SERVICE_CHOICES = {"Taxi + Uber/Lyft": ("yellow", "hvfhv"), "Uber/Lyft": ("hvfhv",), "Yellow taxi": ("yellow",)}

st.set_page_config(page_title="NYC Congestion Toll", page_icon="🚕", layout="wide")


def pct(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}%}"


def metrics(name: str) -> dict:
    return json.loads((METRICS / f"{name}.json").read_text())


@st.cache_data(show_spinner="Loading 1.3 billion trips (as daily zone totals)...")
def zone_day() -> pd.DataFrame:
    return load_zone_day()


@st.cache_data(show_spinner=False)
def main_panel() -> pd.DataFrame:
    """The notebook-03 panel: CBD zones vs the rest of Manhattan, taxi + Uber/Lyft."""
    p = zone_week_panel(zone_day(), ("yellow", "hvfhv"))
    return p[p["pu_zone"].map(zone_group).isin(["cbd", "manhattan_north"])].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def weekly_by_group(services: tuple[str, ...]) -> pd.DataFrame:
    zd = zone_day()
    df = zd[zd["service"].isin(services) & zd["group"].isin(list(GROUP_LABEL))].copy()
    df["trips_out"] = np.where(df["group"] == "cbd", df["trips"], df["trips"] - df["trips_to_cbd"])
    df["week"] = week_start(df["date"])
    w = df.groupby(["week", "group"])["trips_out"].sum().unstack()
    days = df.groupby("week")["date"].nunique()
    w = w[(days == 7) & (w.index >= "2022-01-03") & (w.index != pd.Timestamp("2024-12-30"))]
    return w


@st.cache_data(show_spinner=False)
def zone_change() -> pd.DataFrame:
    """Trips per pickup zone, 2025 vs 2024 (taxi + Uber/Lyft)."""
    zd = zone_day()
    df = zd[zd["service"].isin(["yellow", "hvfhv"])]
    y = df.assign(year=df["date"].dt.year).query("year in (2024, 2025)")
    t = y.pivot_table(index="pu_zone", columns="year", values="trips", aggfunc="sum").fillna(0)
    t = t[t[2024] >= 5000]
    return pd.DataFrame({"zone": t.index, "trips_2024": t[2024].values, "trips_2025": t[2025].values,
                         "change": (t[2025] / t[2024] - 1).values})


@st.cache_data(show_spinner=False)
def zone_options() -> pd.DataFrame:
    """Pickup zones busy enough to explore (at least 500 taxi + Uber/Lyft trips a week in 2024)."""
    zd = zone_day()
    y = zd[zd["service"].isin(["yellow", "hvfhv"]) & (zd["date"].dt.year == 2024)]
    weekly = y.groupby("pu_zone")["trips"].sum() / 52
    lookup = pd.read_csv(ROOT / "data" / "reference" / "taxi_zone_lookup.csv").set_index("LocationID")
    out = pd.DataFrame({"zone": weekly.index, "weekly_2024": weekly.values})
    out = out[(out["weekly_2024"] >= 500) & out["zone"].map(zone_group).isin(list(GROUP_LABEL))]
    out["name"] = out["zone"].map(lookup["Zone"])
    out["borough"] = out["zone"].map(lookup["Borough"])
    out["group"] = out["zone"].map(zone_group)
    out["label"] = (out["name"] + " (" + out["borough"] + ")"
                    + np.where(out["group"] == "cbd", " · toll zone", ""))
    return out.sort_values("weekly_2024", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def zone_weekly(zone: int, services: tuple[str, ...]) -> pd.DataFrame:
    zd = zone_day()
    df = zd[(zd["pu_zone"] == zone) & zd["service"].isin(services)].copy()
    df["week"] = week_start(df["date"])
    w = df.groupby("week")[["trips", "miles_sum", "minutes_sum", "fare_sum", "fee_trips",
                            "shared_requests"]].sum()
    days = df.groupby("week")["date"].nunique()
    return w[(days == 7) & (w.index >= "2022-01-03")]


@st.cache_data(show_spinner=False)
def area_year(group: str, services: tuple[str, ...], year: int) -> pd.Series:
    zd = zone_day()
    df = zd[(zd["group"] == group) & zd["service"].isin(services) & (zd["date"].dt.year == year)]
    return df[["trips", "miles_sum", "minutes_sum"]].sum()


def zone_year(zone: int, services: tuple[str, ...], year: int) -> pd.Series:
    zd = zone_day()
    df = zd[(zd["pu_zone"] == zone) & zd["service"].isin(services) & (zd["date"].dt.year == year)]
    return df[["trips", "miles_sum", "minutes_sum", "fare_sum", "fee_trips", "shared_requests"]].sum()


@st.cache_data(show_spinner=False)
def fake_toll(start: str) -> dict:
    """DiD on pre-toll data only, pretending the toll started on `start`."""
    p = main_panel()
    pre = p[p["week"] < pd.Timestamp("2024-12-30")]
    s = pd.Timestamp(start)
    lo = pd.Timestamp("2022-07-04")                  # same design as notebook 03, section 4b
    hi = min(s + pd.Timedelta(weeks=51), pre["week"].max())
    win = window(pre, lo, hi, s)
    quick = fit_did(win, weights="pre_mean_trips")
    trend = fit_did(win, weights="pre_mean_trips", unit_trends=True)
    return {"quick": quick.pct, "quick_ci": quick.pct_ci, "trend": trend.pct, "trend_ci": trend.pct_ci,
            "weeks_before": int((s - lo).days / 7), "weeks_after": int((hi - s).days / 7) + 1}


did, fc, sc, summ = metrics("did"), metrics("forecast"), metrics("synth"), metrics("summary")
audit = metrics("data_audit")
methods = pd.read_csv(METRICS / "methods_comparison.csv")
other = pd.read_csv(METRICS / "did_other_outcomes.csv")

# --- Header -------------------------------------------------------------------
st.title("🚕 Did NYC's Congestion Toll Cut Taxi and Uber/Lyft Trips?")
st.markdown(
    "Since **5 January 2025**, vehicles entering Manhattan below 60th Street pay a toll, and every "
    "taxi trip in the zone carries a **\\$0.75** fee (Uber/Lyft: **\\$1.50**). This app shows what "
    f"happened to trips starting in the zone, using **{summ['trips_kept'] / 1e9:.2f} billion** TLC "
    "trip records from January 2022 to August 2026."
)
with st.expander("ℹ️ How to use this demo"):
    st.markdown(
        """
1. **🏁 The answer**: the result, the three methods side by side, and what the toll did change.
2. **📈 Trends**: why a simple before/after comparison misleads: the toll zone was already losing
   share of trips before the toll.
3. **🔎 Explore a zone**: pick any neighbourhood and see its own trips, speed and fees.
4. **🔮 Forecast vs actual**: the zone's trips against a forecast built only from pre-toll data.
5. **🧪 Try a fake toll date**: run the same analysis on a date when nothing happened, to see how
   much "effect" each method finds by chance.

**CBD** = the toll zone (Manhattan's Central Business District). **DiD** = difference-in-differences:
how much more the CBD changed than a comparison area.
"""
    )

tab_answer, tab_trends, tab_zone, tab_fc, tab_fake = st.tabs(
    ["🏁 The answer", "📈 Trends", "🔎 Explore a zone", "🔮 Forecast vs actual", "🧪 Try a fake toll date"])

# =====================================================================
# TAB 1 -- THE ANSWER
# =====================================================================
with tab_answer:
    main = did["main_trend_adjusted_year1"]
    st.success(
        f"**No detectable effect on the number of trips in year one.** Allowing for the decline that "
        f"started before the toll, trips from the CBD changed by **{pct(main['effect'])}** "
        f"(95% CI {pct(main['ci_low'])} to {pct(main['ci_high'])}). What the toll did change is how "
        "trips run and which rides people choose."
    )
    st.markdown("#### The answer in four numbers")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trips from the toll zone", pct(main["effect"]), border=True,
              help="Main estimate: each zone keeps its own 2023-2024 trend, and the toll effect is the "
                   "break from it. Not distinguishable from zero.")
    k2.metric("Uber/Lyft trip speed", pct(summ["speed_uber"]), border=True,
              help=f"Yellow taxis: {pct(summ['speed_taxi'])}. Both are well beyond what the method finds "
                   "on fake toll dates.")
    k3.metric("Uber/Lyft driver pay per hour on a trip", pct(summ["pay_per_hour_uber"]), border=True,
              help="The same trips take less time, and pay is partly per trip.")
    k4.metric("Uber/Lyft shared-ride requests", pct(summ["shared_requests_uber"]), border=True,
              help="The fee has no discount for sharing, so it weighs most on the cheapest rides.")

    st.markdown("#### Why a simple before/after comparison gets it wrong")
    q1, q2, q3 = st.columns([1, 1, 2])
    q1.metric("2025 vs 2024", pct(did["quick_2025_vs_2024"]["effect"]), border=True,
              help="CBD trips against the rest of Manhattan, ignoring earlier trends.")
    q2.metric("2024 vs 2023 (no toll yet)", pct(did["placebo_2024_vs_2023"]["effect"]), border=True,
              help="The same comparison a year earlier: the gap was already there.")
    q3.info("The toll zone was **already losing share of trips** (about 4% a year) before the toll. "
            "A quick 2025-vs-2024 comparison reads that decline as a toll effect; the same comparison "
            "a year earlier, with no toll, shows an even bigger \"drop\". See the **📈 Trends** tab.")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Every method, side by side**")
        m = methods.copy()
        m["kind"] = np.where(m["method"].str.contains("no toll"), "No-toll placebo",
                             np.where(m["accounts_for_trend"], "Accounts for the pre-toll trend",
                                      "Ignores the pre-toll trend"))
        m["label"] = m["method"] + np.where(m["note"].fillna("").str.startswith("placebo p"),
                                            " (" + m["note"].fillna("") + ")", "")
        order = m["label"].tolist()
        base = alt.Chart(m).encode(y=alt.Y("label:N", sort=order, title=None,
                                           axis=alt.Axis(labelLimit=330, labelOverlap=False)))
        color = alt.Color("kind:N", title=None, legend=alt.Legend(orient="bottom", columns=1, labelLimit=400),
                          scale=alt.Scale(domain=["Ignores the pre-toll trend", "No-toll placebo",
                                                  "Accounts for the pre-toll trend"],
                                          range=["#94A3B8", "#A16207", COLORS["cbd"]]))
        rules = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q", color=color)
        pts = base.mark_circle(size=110).encode(
            x=alt.X("effect:Q", title="Effect on CBD trips, 2025", axis=alt.Axis(format="+%")),
            color=color, tooltip=["method", alt.Tooltip("effect:Q", format="+.1%"),
                                  alt.Tooltip("low:Q", format="+.1%"), alt.Tooltip("high:Q", format="+.1%"),
                                  "note"])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")
        st.altair_chart((zero + rules + pts).properties(height=380), width="stretch")
        st.caption("Grey and brown: methods that ignore the pre-toll slide, and the no-toll year. Orange: "
                   "methods that account for it; all land near zero. None can rule out a small effect: on "
                   f"fake toll dates the main method finds about ±{did['fake_dates']['trend_adjusted_rms']:.0%} "
                   "(see the 🧪 tab).")
    with right:
        st.markdown("**What the toll did change** (trend-adjusted, 2025)")
        show_all = st.toggle("Show every outcome tested", value=False, key="show_all_outcomes",
                             help="Including the ones that move about as much on fake toll dates.")
        o = other[other["outcome"] != "Trips (main estimate)"].copy()
        if not show_all:
            o = o[o["clear"]]
        o["label"] = o["service"] + " · " + o["outcome"]
        o["verdict"] = np.where(o["clear"], "Clearly beyond fake-date effects", "Not clearly beyond them")
        bars = alt.Chart(o).mark_bar().encode(
            y=alt.Y("label:N", sort=o["label"].tolist(), title=None,
                    axis=alt.Axis(labelLimit=330, labelOverlap=False)),
            x=alt.X("effect:Q", title="Change", axis=alt.Axis(format="+%")),
            color=alt.Color("verdict:N", title=None, legend=alt.Legend(orient="bottom", columns=1, labelLimit=400),
                            scale=alt.Scale(domain=["Clearly beyond fake-date effects", "Not clearly beyond them"],
                                            range=[COLORS["cbd"], "#CBD5E1"])),
            tooltip=["label", alt.Tooltip("effect:Q", format="+.1%"),
                     alt.Tooltip("fake_rms:Q", format=".1%", title="Typical fake-date effect"),
                     alt.Tooltip("placebo_2024:Q", format="+.1%", title="Fake January 2024 toll")])
        st.altair_chart(bars.properties(height=330 if not show_all else 560), width="stretch")
        st.caption("Each outcome counts only if it is at least twice what the same method finds on fake "
                   "toll dates. Turn on the switch to see the ones that did not pass (Uber/Lyft waits came "
                   "close).")

    st.markdown("#### What it means for pricing")
    el = summ["elasticity"]
    p1, p2, p3 = st.columns(3)
    p1.metric("Fee as a share of a CBD trip's fare", f"{el['price_increase']:.1%}", border=True,
              help="Per-trip fee divided by the fare before fees and tips, CBD trips in 2025.")
    p2.metric("Price elasticity: strongest still possible", f"{el['forecast_elasticity_bound']:.1f}", border=True,
              help=f"From the forecast check: trips fell by at most {abs(el['forecast_lowest_trip_change']):.1%} "
                   "for a price rise of this size.")
    p3.metric("CBD fees paid by riders, 2025", f"${audit['fees_2025_usd'] / 1e6:,.1f}M", border=True,
              help="Taxi and Uber/Lyft per-trip fees recorded in the TLC data.")
    st.caption("A 4% price rise cost at most about 2% of trips: demand in the zone is not very sensitive to "
               "a fee of this size. The one place a targeted offer is worth testing is shared rides.")
    st.info(
        f"**What to watch:** in January-August 2026 CBD trips are {pct(summ['year2_did'])} against "
        f"zone trends, but the rest of Manhattan fell short of its forecast by about as much "
        f"(CBD minus Manhattan: {pct(summ['year2_forecast_cbd_minus_manhattan'])}), so the drop "
        "looks Manhattan-wide. "
        f"[Decision memo]({REPO_URL}/blob/main/reports/decision_memo.md)"
    )

# =====================================================================
# TAB 2 -- TRENDS
# =====================================================================
with tab_trends:
    svc_name = st.radio("Service", list(SERVICE_CHOICES), horizontal=True, key="svc")
    w = weekly_by_group(SERVICE_CHOICES[svc_name])
    base_2023 = w[(w.index >= "2023-01-01") & (w.index < "2024-01-01")].mean()
    idx = (w / base_2023 * 100).rolling(4).mean().reset_index().melt(
        id_vars="week", var_name="group", value_name="index")
    idx["area"] = idx["group"].map(GROUP_LABEL)
    line = alt.Chart(idx.dropna()).mark_line(strokeWidth=2).encode(
        x=alt.X("week:T", title=None),
        y=alt.Y("index:Q", title="Weekly trips (2023 average = 100)", scale=alt.Scale(zero=False)),
        color=alt.Color("area:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=list(GROUP_LABEL.values()), range=list(COLORS.values()))),
        tooltip=["area", alt.Tooltip("week:T"), alt.Tooltip("index:Q", format=".1f")])
    toll = alt.Chart(pd.DataFrame({"x": [TOLL_DAY]})).mark_rule(strokeDash=[5, 4], color="black").encode(x="x:T")
    st.markdown("**Trips by area** (4-week average; comparison areas exclude trips into the CBD)")
    st.altair_chart((line + toll).properties(height=300), width="stretch")
    st.caption("The CBD (orange) was already growing more slowly than the rest of the city in 2023 and "
               "2024. A before/after comparison in 2025 picks up that gap and calls it the toll.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**The CBD's share against the rest of Manhattan**")
        r = np.log(w["cbd"] / w["manhattan_north"])
        r = (r - r[(r.index >= "2024-01-01") & (r.index < "2024-12-30")].mean()).rolling(4).mean()
        pre = r[(r.index >= "2023-01-09") & (r.index < "2024-12-30")].dropna()
        t = (pre.index - pre.index[0]).days / 365.25
        b, a = np.polyfit(t, pre.values, 1)
        tt = (r.index - pre.index[0]).days / 365.25
        share = pd.DataFrame({"week": r.index, "Actual": np.expm1(r.values),
                              "2023-24 trend, extended": np.where(r.index >= "2023-01-09",
                                                                   np.expm1(a + b * tt), np.nan)})
        share = share.melt(id_vars="week", var_name="what", value_name="gap").dropna()
        ch = alt.Chart(share).mark_line().encode(
            x=alt.X("week:T", title=None), y=alt.Y("gap:Q", title="vs 2024 average", axis=alt.Axis(format="+%")),
            color=alt.Color("what:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=["Actual", "2023-24 trend, extended"],
                                            range=[COLORS["cbd"], "#111827"])),
            strokeDash=alt.StrokeDash("what:N", legend=None,
                                      scale=alt.Scale(domain=["Actual", "2023-24 trend, extended"],
                                                      range=[[1, 0], [4, 4]])))
        st.altair_chart((ch + toll).properties(height=280), width="stretch")
        st.caption("Before the toll the CBD was steadily losing share (dotted line). Through 2025 it "
                   "stays close to that line; it drops below it in early 2026.")
    with c2:
        st.markdown("**Event study** (taxi + Uber/Lyft, quarterly)")
        es = pd.read_csv(METRICS / "event_study.csv", parse_dates=["start_week"])
        pts = alt.Chart(es).mark_circle(size=70, color=COLORS["cbd"]).encode(
            x=alt.X("start_week:T", title=None),
            y=alt.Y("pct:Q", title="CBD gap vs the quarter before the toll", axis=alt.Axis(format="+%")),
            tooltip=[alt.Tooltip("start_week:T", title="Quarter from"), alt.Tooltip("pct:Q", format="+.1%"),
                     alt.Tooltip("pct_ci_low:Q", format="+.1%"), alt.Tooltip("pct_ci_high:Q", format="+.1%")])
        err = alt.Chart(es).mark_rule(color="#FDBA74").encode(x="start_week:T", y="pct_ci_low:Q",
                                                             y2="pct_ci_high:Q")
        trend_line = alt.Chart(es).mark_line(strokeDash=[2, 3], color="#111827").encode(
            x="start_week:T", y="pre_trend:Q")
        st.altair_chart((err + pts + trend_line + toll).properties(height=280), width="stretch")
        st.caption("Before the toll the gap slopes down (dotted line), so the parallel-trends "
                   "assumption of a plain DiD fails. The 2025 points continue that slope.")

    st.markdown("**Change in trips by pickup zone, 2025 vs 2024** (taxi + Uber/Lyft)")
    geo = json.loads((ROOT / "data" / "reference" / "taxi_zones.geojson").read_text())
    ch_df = zone_change().set_index("zone")
    for f in geo["features"]:
        z = int(f["properties"]["LocationID"])
        f["properties"]["change"] = float(ch_df["change"].get(z, np.nan)) if z in ch_df.index else None
        f["properties"]["in_cbd"] = "Toll zone" if z in CBD_ZONES else "Outside"
    feats = alt.Data(values=[f for f in geo["features"] if f["properties"]["borough"] != "EWR"])
    zmap = alt.Chart(feats).mark_geoshape(stroke="white", strokeWidth=0.3).encode(
        color=alt.Color("properties.change:Q", title="2025 vs 2024",
                        scale=alt.Scale(scheme="redblue", domain=[-0.3, 0.3], clamp=True),
                        legend=alt.Legend(format="+%")),
        tooltip=[alt.Tooltip("properties.zone:N", title="Zone"),
                 alt.Tooltip("properties.borough:N", title="Borough"),
                 alt.Tooltip("properties.in_cbd:N", title="Area"),
                 alt.Tooltip("properties.change:Q", title="Change", format="+.1%")],
    ).project("mercator")
    outline = alt.Chart(alt.Data(values=[f for f in geo["features"]
                                         if int(f["properties"]["LocationID"]) in CBD_ZONES])).mark_geoshape(
        fill=None, stroke="black", strokeWidth=1.2).project("mercator")
    st.altair_chart((zmap + outline).properties(height=520), width="stretch")
    st.caption("Black outlines: the 38 toll-zone pickup zones. Zones with fewer than 5,000 trips in 2024 "
               "are blank. Hover a zone for its change.")

# =====================================================================
# TAB 3 -- EXPLORE A ZONE
# =====================================================================
with tab_zone:
    st.markdown(
        "Pick any busy pickup zone to see its own trips, speeds and fees before and after the toll. "
        "These are plain before/after numbers for one neighbourhood: they still contain the citywide "
        "trends that the methods in the other tabs remove."
    )
    opts = zone_options()
    zc1, zc2 = st.columns([2, 1])
    choice = zc1.selectbox("Pickup zone (busiest first)", opts["label"].tolist(), index=0, key="zone_pick")
    zsvc_name = zc2.radio("Service", list(SERVICE_CHOICES), horizontal=True, key="zone_svc")
    zrow = opts[opts["label"] == choice].iloc[0]
    zone_id, zsvc = int(zrow["zone"]), SERVICE_CHOICES[zsvc_name]
    y24, y25 = zone_year(zone_id, zsvc, 2024), zone_year(zone_id, zsvc, 2025)

    def ratio(a, b):
        return (a / b - 1) if b else float("nan")

    def mph(y):
        return y["miles_sum"] / (y["minutes_sum"] / 60) if y["minutes_sum"] else float("nan")

    ref24, ref25 = area_year("manhattan_north", zsvc, 2024), area_year("manhattan_north", zsvc, 2025)
    ref_trips, ref_mph = ratio(ref25["trips"], ref24["trips"]), ratio(mph(ref25), mph(ref24))
    z1, z2, z3, z4 = st.columns(4)
    z1.metric("Trips, 2025 vs 2024", pct(ratio(y25["trips"], y24["trips"])), border=True,
              help=f"{y25['trips']:,.0f} trips in 2025 against {y24['trips']:,.0f} in 2024.")
    z2.metric("Average trip speed, 2025 vs 2024", pct(ratio(mph(y25), mph(y24))), border=True,
              help=f"{mph(y25):.1f} mph in 2025 against {mph(y24):.1f} mph in 2024, over the whole trip "
                   "(which may run well outside the zone).")
    z3.metric("Trips paying the CBD fee, 2025", f"{y25['fee_trips'] / y25['trips']:.0%}" if y25["trips"] else "n/a",
              border=True, help="Trips that start in the toll zone, or end in or pass through it, pay the fee.")
    if "hvfhv" in zsvc and y24["shared_requests"] >= 200:
        z4.metric("Shared-ride requests, 2025 vs 2024",
                  pct(ratio(y25["shared_requests"], y24["shared_requests"])), border=True,
                  help="Uber/Lyft only.")
    else:
        z4.metric("Average fare, 2025 vs 2024",
                  pct(ratio(y25["fare_sum"] / y25["trips"], y24["fare_sum"] / y24["trips"])), border=True,
                  help="Fare before fees and tips.")

    st.caption(f"For comparison, Manhattan above the toll zone: trips {pct(ref_trips)}, speed {pct(ref_mph)} "
               "(same service, 2025 vs 2024).")

    zw = zone_weekly(zone_id, zsvc)
    area = weekly_by_group(zsvc)[zrow["group"]]
    AREA_NAME = {"cbd": "Whole toll zone", "manhattan_north": "Whole rest of Manhattan",
                 "outer": "All outer boroughs"}
    zone_name, area_name = str(zrow["name"]), AREA_NAME[zrow["group"]]
    both = pd.DataFrame({zone_name: zw["trips"], area_name: area})
    base24 = both[(both.index >= "2024-01-01") & (both.index < "2024-12-30")].mean()
    both = (both / base24 * 100).rolling(4).mean().reset_index(names="week").melt(
        id_vars="week", var_name="series", value_name="index").dropna()
    zline = alt.Chart(both).mark_line(strokeWidth=2).encode(
        x=alt.X("week:T", title=None),
        y=alt.Y("index:Q", title="Weekly trips (2024 average = 100)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom", labelLimit=400),
                        scale=alt.Scale(domain=[zone_name, area_name],
                                        range=[COLORS["cbd"] if zrow["group"] == "cbd" else COLORS["manhattan_north"],
                                               "#94A3B8"])),
        tooltip=["series", alt.Tooltip("week:T"), alt.Tooltip("index:Q", format=".1f")])
    st.altair_chart((zline + toll).properties(height=320), width="stretch")
    st.caption(f"{zone_name} against {area_name.lower()}, both set to 100 for the "
               "2024 average, 4-week rolling. The dashed line is the toll's first day. A zone that follows its "
               "area through the toll date moved with the city, not with the toll.")

# =====================================================================
# TAB 4 -- FORECAST VS ACTUAL
# =====================================================================
with tab_fc:
    st.markdown(
        "A second way to ask the question: forecast each area's daily trips from **pre-toll data only** "
        "(July 2022–December 2024; model: regression with weather and holidays, plus LightGBM), then "
        "compare with what happened. The bands come from backtests on years with no toll."
    )
    series_name = st.radio("Area", ["Toll zone (CBD)", "Rest of Manhattan"], horizontal=True, key="fc_series")
    key = "cbd" if series_name.startswith("Toll") else "manhattan"
    fw = pd.read_csv(METRICS / "forecast_weekly.csv", parse_dates=["week"])
    fw = fw[fw["series"] == key]
    m1, m2, m3 = st.columns(3)
    m1.metric("2025: actual vs forecast", pct(fc["gap_2025"][key]), border=True,
              help=f"Largest gap in the no-toll backtests: ±{fc['placebo_noise_max_abs']:.1%}. Net of the "
                   f"average backtest gap: {pct(fc['gap_2025_net_of_backtest'][key])}.")
    m2.metric("Jan–Aug 2026: actual vs forecast", pct(fc["gap_2026"][key]), border=True)
    m3.metric("Forecast error (CBD, weekly)", f"{fc['cbd_weekly_mape_chosen']:.1%}", border=True,
              help="Mean absolute percentage error on weekly trips in backtests with no toll.")
    band95 = alt.Chart(fw.dropna(subset=["forecast"])).mark_area(color="#FDBA74", opacity=0.3).encode(
        x="week:T", y=alt.Y("lo95:Q", title="Trips per week", scale=alt.Scale(zero=False)), y2="hi95:Q")
    band80 = alt.Chart(fw.dropna(subset=["forecast"])).mark_area(color="#FDBA74", opacity=0.45).encode(
        x="week:T", y="lo80:Q", y2="hi80:Q")
    fc_line = alt.Chart(fw.dropna(subset=["forecast"])).mark_line(strokeDash=[5, 3], color="#111827").encode(
        x="week:T", y="forecast:Q",
        tooltip=[alt.Tooltip("week:T"), alt.Tooltip("forecast:Q", format=",.0f"),
                 alt.Tooltip("actual:Q", format=",.0f")])
    act = alt.Chart(fw).mark_line(color=COLORS["cbd"] if key == "cbd" else COLORS["manhattan_north"]).encode(
        x=alt.X("week:T", title=None),
        y=alt.Y("actual:Q", title="Trips per week", scale=alt.Scale(zero=False),
                axis=alt.Axis(labelExpr="format(datum.value / 1e6, '.2f') + 'M'")),
        tooltip=[alt.Tooltip("week:T"), alt.Tooltip("actual:Q", format=",.0f")])
    st.altair_chart((band95 + band80 + fc_line + act + toll).properties(height=360), width="stretch")
    st.caption(
        "Solid: actual weekly trips. Dashed: the no-toll forecast. Shading: 80% and 95% bands. "
        f"In 2025 the CBD stayed inside its band ({fc['weeks_outside_95_band']:.0%} of weeks outside the "
        "95% band). In 2026 both areas fall below their forecasts by a similar amount."
    )

# =====================================================================
# TAB 5 -- TRY A FAKE TOLL DATE
# =====================================================================
with tab_fake:
    st.markdown(
        "Pick a date **before** the real toll. The app runs the same two DiD models on data from July 2022 "
        "to a year after that date, all before January 2025. Nothing happened on these dates, so whatever "
        "\"effect\" appears is what each method finds by chance or by trend."
    )
    mondays = pd.date_range("2023-07-03", "2024-01-01", freq="W-MON")
    start = st.select_slider("Fake toll date", options=[d.date().isoformat() for d in mondays],
                             value="2023-10-02", key="fake_start")
    with st.spinner("Fitting both models..."):
        r = fake_toll(start)
    f1, f2, f3 = st.columns(3)
    f1.metric("Quick DiD (no trends)", pct(r["quick"]), border=True,
              help=f"95% CI {pct(r['quick_ci'][0])} to {pct(r['quick_ci'][1])}")
    f2.metric("Trend-adjusted DiD", pct(r["trend"]), border=True,
              help=f"95% CI {pct(r['trend_ci'][0])} to {pct(r['trend_ci'][1])}")
    f3.metric("Real toll, trend-adjusted", pct(did["main_trend_adjusted_year1"]["effect"]), border=True,
              help="The main estimate, for comparison.")
    st.caption(f"Window: {r['weeks_before']} weeks before and {r['weeks_after']} weeks after the fake date.")

    fd = pd.read_csv(METRICS / "did_fake_dates.csv", parse_dates=["fake_start"]).melt(
        id_vars="fake_start", var_name="model", value_name="effect")
    fd["model"] = fd["model"].map({"quick": "Quick DiD", "trend_adjusted": "Trend-adjusted DiD",
                                   "pre_trend_two_step": "Two-step (pre-toll trends only)"})
    fake_chart = alt.Chart(fd).mark_line(point=True).encode(
        x=alt.X("fake_start:T", title="Fake toll date"),
        y=alt.Y("effect:Q", title="\"Effect\" found", axis=alt.Axis(format="+%")),
        color=alt.Color("model:N", title=None, legend=alt.Legend(orient="bottom", labelLimit=400),
                        scale=alt.Scale(domain=["Quick DiD", "Trend-adjusted DiD", "Two-step (pre-toll trends only)"],
                                        range=["#94A3B8", COLORS["cbd"], "#A16207"])),
        tooltip=[alt.Tooltip("fake_start:T"), "model", alt.Tooltip("effect:Q", format="+.1%")])
    you = alt.Chart(pd.DataFrame({"x": [pd.Timestamp(start)]})).mark_rule(color="#111827", strokeDash=[3, 3]).encode(
        x="x:T")
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q")
    st.markdown("**Seven fake dates, four weeks apart** (from notebook 03)")
    st.altair_chart((zero_line + fake_chart + you).properties(height=280), width="stretch")
    st.info(
        "The quick DiD finds a drop on every fake date: it reads the CBD's ongoing decline as an effect. "
        "The trend-adjusted DiD has no such bias but scatters by a few percent, so on its own it cannot "
        "rule out a small real effect. That is why the forecast and synthetic-control checks matter: "
        "all three land near zero for the real toll."
    )

st.divider()
st.caption(
    "Data: NYC Taxi & Limousine Commission trip records (public), Jan 2022–Aug 2026. Methods: "
    f"`src/congestion/` (tested). [Source on GitHub]({REPO_URL}) · "
    f"[Decision memo]({REPO_URL}/blob/main/reports/decision_memo.md)"
)
