"""Streamlit dashboard. Reads only saved outputs (reports/benchmark/, data/raw/traffic_api/).

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_settings  # noqa: E402
from src.utils.raw_io import read_raw_run  # noqa: E402

BENCH = ROOT / "reports" / "benchmark"
LIVE_DIR = ROOT / "data" / "raw" / "traffic_api"
CLASSES = ["Low", "Moderate", "Severe"]
COLORS = {"Low": "#2e9e4f", "Moderate": "#f28c28", "Severe": "#d62728"}

st.set_page_config(page_title="Traffic Congestion Prediction", layout="wide")


# --- loaders (cached; they only read saved files) ------------------------------------------

@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(BENCH / "metrics.csv")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BENCH / name)


@st.cache_data
def load_shap() -> pd.DataFrame | None:
    path = BENCH / "shap_top_features.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_routing_meta() -> dict:
    path = BENCH / "routing_meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_live_snapshot() -> tuple[int, str | None, pd.DataFrame | None]:
    """(number of raw run files, latest run name, latest snapshot per segment)."""
    files = sorted(LIVE_DIR.glob("*/run_*.json*"), key=lambda p: p.name)
    if not files:
        return 0, None, None
    latest = files[-1]
    rows = []
    for rec in read_raw_run(latest):
        seg = rec["response"]["flowSegmentData"]
        cur, free = seg.get("currentSpeed"), seg.get("freeFlowSpeed")
        rows.append({
            "segment": rec["segment_key"], "currentSpeed (km/h)": cur, "freeFlowSpeed (km/h)": free,
            "speed_ratio": round(cur / free, 2) if cur is not None and free else None,
            "confidence": seg.get("confidence"),
        })
    return len(files), latest.name.split(".")[0], pd.DataFrame(rows)


def parse_dq_rows(md: str) -> pd.DataFrame:
    """Pull the `| item | count | pct% |` rows out of dq_summary.md for the chart."""
    rows = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and re.fullmatch(r"\d+(\.\d+)?%", cells[2]):
            count = cells[1].replace(",", "")
            rows.append({
                "item": cells[0].replace("`", ""), "count": int(count) if count.isdigit() else None,
                "pct": float(cells[2].rstrip("%")),
            })
    return pd.DataFrame(rows)


def final_model_name() -> str:
    """Final model recorded by the training run (falls back to XGBoost)."""
    path = BENCH / "final_model.json"
    return json.loads(path.read_text())["model"] if path.exists() else "xgboost"


def missing_notice(path: Path, hint: str) -> None:
    st.info(f"`{path.relative_to(ROOT)}` not found yet. {hint}")


# --- tabs -----------------------------------------------------------------------------------

def tab_overview() -> None:
    st.header("Congestion prediction & route optimization")
    st.markdown(
        "Predict congestion level (**Low / Moderate / Severe**, from `speed_ratio = current speed / free-flow speed`) "
        "for road segments **30 minutes ahead**, then recommend congestion-aware routes versus plain "
        "shortest-distance routes. The **primary live source** is the Bengaluru TomTom Traffic Flow API (South-East corridor); "
        "because no public historical Indian sensor data exists, the model is validated on the public **METR-LA** "
        "benchmark (the **benchmark training source**) and will be retrained on the Bengaluru data as it accumulates."
    )

    st.subheader("Pipeline flow")
    st.graphviz_chart(
        "digraph { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica]; "
        'P [label="Problem"]; S [label="Sources"]; I [label="Ingestion\\n(retry + quarantine)"]; '
        'R [label="Raw zones\\n(append-only,\\nmanifests)"]; T [label="Transform\\n(DQ rules, flags)"]; '
        'G [label="Integrate\\n(join keys)"]; C [label="Curated\\nparquet"]; M [label="ML\\n(model, SHAP, routing)"]; '
        "P -> S -> I -> R -> T -> G -> C -> M }"
    )

    st.subheader("Data sources")
    sources_path = BENCH / "sources.csv"
    if sources_path.exists():
        st.dataframe(load_csv("sources.csv"), hide_index=True, width="stretch")
    else:
        missing_notice(sources_path, "Run `python -m src.benchmark.manifests`.")
    st.markdown(
        "**Schema differences:** METR-LA is a wide HDF5 matrix (timestamp x 207 sensor columns) plus CSVs; Open-Meteo is "
        "column-oriented JSON arrays with ISO time strings; holidays are a two-column CSV. They differ in granularity "
        "(5 min / 1 h / 1 day), so each joins on its own key."
    )
    schemas = BENCH / "source_schemas.md"
    if schemas.exists():
        with st.expander("Per-source columns, integration keys and contributed features"):
            st.markdown(schemas.read_text(encoding="utf-8"))

    st.subheader("Bengaluru live collection")
    n_files, latest, snap = load_live_snapshot()
    if snap is None:
        st.info("No snapshots collected yet in `data/raw/traffic_api/`.")
        return
    c1, c2 = st.columns(2)
    c1.metric("Raw snapshot files", n_files)
    c2.metric("Latest snapshot", latest.replace("run_", ""))
    st.dataframe(snap, hide_index=True, width="stretch")


# --- Bengaluru live congestion ----------------------------------------------------------------

def congestion_level(ratio: float, thresholds: dict[str, float]) -> str:
    """Low >= low, Moderate in [severe, low), Severe < severe (thresholds from config/settings.yaml)."""
    if ratio >= thresholds["low"]:
        return "Low"
    return "Moderate" if ratio >= thresholds["severe"] else "Severe"


@st.cache_data(ttl=300)
def load_live_history() -> pd.DataFrame:
    """Every raw TomTom run joined with the segment metadata; one row per (snapshot, segment)."""
    settings = get_settings()
    thresholds = settings["thresholds"]
    seg = pd.DataFrame(settings["segments"]).set_index("segment_key")
    rows = []
    for path in sorted(LIVE_DIR.glob("*/run_*.json*"), key=lambda p: p.name):
        stamp = pd.to_datetime(path.name.split(".")[0].removeprefix("run_"), format="%Y%m%dT%H%MZ", utc=True)
        for rec in read_raw_run(path):
            flow = rec["response"]["flowSegmentData"]
            cur, free = flow.get("currentSpeed"), flow.get("freeFlowSpeed")
            if not free or cur is None or rec["segment_key"] not in seg.index:
                continue
            meta = seg.loc[rec["segment_key"]]
            ratio = cur / free
            rows.append({
                "snapshot_utc": stamp, "snapshot_ist": stamp.tz_convert("Asia/Kolkata"), "segment": rec["segment_key"],
                "name": meta["name"], "road": meta["osm_road_name"], "lat": meta["lat"], "lon": meta["lon"],
                "currentSpeed": cur, "freeFlowSpeed": free, "speed_ratio": round(ratio, 3),
                "confidence": flow.get("confidence"), "level": congestion_level(ratio, thresholds),
            })
    return pd.DataFrame(rows)


def tab_live() -> None:
    st.header("Bengaluru live congestion")
    hist = load_live_history()
    if hist.empty:
        st.info("No TomTom snapshots found in `data/raw/traffic_api/` yet.")
        return
    snaps = sorted(hist["snapshot_ist"].unique(), reverse=True)           # newest first = default
    choice = st.selectbox("Snapshot (IST)", snaps, format_func=lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M IST"))
    now = hist[hist["snapshot_ist"] == choice]

    counts = now["level"].value_counts().reindex(["Low", "Moderate", "Severe"], fill_value=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments", len(now))
    c2.metric("Low", int(counts["Low"]))
    c3.metric("Moderate", int(counts["Moderate"]))
    c4.metric("Severe", int(counts["Severe"]))

    fig = px.scatter_map(
        now, lat="lat", lon="lon", color="level", color_discrete_map=COLORS, category_orders={"level": CLASSES},
        hover_name="name",
        hover_data={"road": True, "currentSpeed": True, "freeFlowSpeed": True, "speed_ratio": True, "confidence": True,
                    "lat": False, "lon": False, "level": False},
        center={"lat": float(now["lat"].mean()), "lon": float(now["lon"].mean())},
        zoom=12, height=600, map_style="open-street-map",
    )
    fig.update_traces(marker={"size": 14})
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend_title_text="Congestion")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Segments at this snapshot")
    table = now[["name", "road", "currentSpeed", "freeFlowSpeed", "speed_ratio", "level", "confidence"]]
    st.dataframe(table.sort_values("speed_ratio"), hide_index=True, width="stretch")

    st.subheader("Speed ratio over time")
    line = px.line(hist.sort_values("snapshot_ist"), x="snapshot_ist", y="speed_ratio", color="name", markers=True,
                   height=420, labels={"snapshot_ist": "Time (IST)", "speed_ratio": "speed_ratio (current / free-flow)"})
    st.plotly_chart(line, width="stretch")
    st.caption("Live data: TomTom Traffic Flow API, every 15 min (GitHub Actions + cron-job.org).")


def tab_model() -> None:
    st.header("Model validation (METR-LA benchmark)")
    st.info("No public historical Indian traffic-sensor data exists, so the model is validated on the METR-LA benchmark. "
            "It will be retrained on the Bengaluru data being collected.")
    if not (BENCH / "metrics.csv").exists():
        missing_notice(BENCH / "metrics.csv", "Run `python -m src.benchmark.run`.")
        return
    metrics = load_metrics()
    split = st.radio("Split", ["test", "val"], horizontal=True)
    table = metrics[metrics["split"] == split].drop(columns="split").reset_index(drop=True)

    def highlight(row: pd.Series) -> list[str]:
        return ["background-color: #fff3b0; font-weight: 600" if row["model"] == "xgboost" else "" for _ in row]

    num_cols = ["accuracy", "macro_f1", "recall_low", "recall_moderate", "recall_severe"]
    st.dataframe(table.style.apply(highlight, axis=1).format({c: "{:.3f}" for c in num_cols}),
                 hide_index=True, width="stretch")
    st.caption("XGBoost (class-weighted) is the final model: highest Severe recall. Random forest is kept for comparison.")

    final = final_model_name()
    c1, c2 = st.columns(2)
    cm = BENCH / "confusion_matrix.png"
    if cm.exists():
        c1.image(str(cm), caption=f"Confusion matrix - {final}, test set")
    else:
        c1.info("confusion_matrix.png not found.")
    shap_df = load_shap()
    if shap_df is not None:
        top = shap_df.head(10).iloc[::-1]
        c2.plotly_chart(px.bar(top, x="mean_abs_shap_severe", y="feature", orientation="h", height=420,
                               title=f"Top features - mean |SHAP|, Severe class ({final})"),
                        width="stretch")

    abl_path = BENCH / "ablation.csv"
    if abl_path.exists():
        st.subheader("Ablation: what does each source add?")
        abl = load_csv("ablation.csv")[["variant", "removed_features", "n_features", "test_accuracy", "test_macro_f1",
                                        "test_recall_severe", "delta_macro_f1", "delta_recall_severe"]]
        st.dataframe(abl.style.format({c: "{:.4f}" for c in abl.columns if c.startswith(("test_", "delta_"))}),
                     hide_index=True, width="stretch")
        st.caption("XGBoost retrained with features removed; the final model is unchanged. Weather and holidays do not "
                   "improve the metrics on this benchmark; see RESULTS.md.")
    st.info("Experiment tracking: every model and ablation run is logged to MLflow (`./mlruns`, experiment "
            "`metr_la_benchmark`). Run `mlflow ui` to view experiments.")

    st.subheader("SHAP (2,000 test rows, Severe class)")
    s1, s2 = st.columns(2)
    for col, name in ((s1, "shap_summary.png"), (s2, "shap_bar.png")):
        if (BENCH / name).exists():
            col.image(str(BENCH / name), caption=name)
        else:
            col.info(f"`reports/benchmark/{name}` not generated yet. Run `python -m src.benchmark.explain`.")

    if shap_df is not None:
        names = shap_df["feature"].tolist()
        share = shap_df["share_severe"]
        ratio_share = share[shap_df["feature"].str.startswith("ratio_")].sum()
        time_share = share[shap_df["feature"].isin(["hour_sin", "hour_cos", "wd_sin", "wd_cos", "is_peak", "is_holiday"])].sum()
        st.markdown(
            f"- Top SHAP features for Severe: **{names[0]}**, **{names[1]}**, **{names[2]}** ({share.iloc[:3].sum():.0%} of total mean |SHAP|).\n"
            f"- Recent speed (ratio) features carry {ratio_share:.0%} and time-of-day/week features {time_share:.0%}; "
            "`free_flow_mph` mostly acts as a sensor identifier."
        )


def tab_routing() -> None:
    st.header("Route optimization (METR-LA benchmark)")
    results = BENCH / "routing_results.csv"
    example = BENCH / "routing_example.png"
    if not results.exists():
        missing_notice(results, "Run `python -m src.routing.benchmark_router`.")
        return
    df = pd.read_csv(results, dtype={"origin": str, "destination": str})
    meta = load_routing_meta()
    if meta:
        st.caption(
            f"Forecast origin {meta['forecast_origin']} (weekday 08:00, test set); routes chosen with predicted speeds, "
            f"timed with actual speeds at {meta['evaluated_at']}. Graph: {meta['nodes']} sensors, {meta['edges']} edges."
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean saving", f"{df['saving_pct'].mean():.2f}%")
    c2.metric("Median saving", f"{df['saving_pct'].median():.2f}%")
    c3.metric("Pairs improved", int((df["saving_pct"] > 0).sum()))
    c4.metric("Pairs worse", int((df["saving_pct"] < 0).sum()))
    if example.exists():
        st.image(str(example), caption="Shortest-distance vs congestion-aware route (best of the 50 pairs)")

    st.subheader("O-D pairs")
    pair = st.selectbox("O-D pair", df.index, format_func=lambda i: f"{df.at[i, 'origin']} -> {df.at[i, 'destination']}")
    row = df.loc[pair]
    a, b, c = st.columns(3)
    a.metric("Distance route", f"{row['dist_route_min']:.2f} min")
    b.metric("Congestion-aware route", f"{row['aware_route_min']:.2f} min")
    c.metric("Saving", f"{row['saving_pct']:.2f}%")
    st.dataframe(df, hide_index=True, width="stretch")


def tab_dq() -> None:
    st.header("Data quality (METR-LA benchmark)")
    rules_path, missing_path = BENCH / "dq_rules.csv", BENCH / "missing_by_source.csv"
    if rules_path.exists():
        st.subheader("DQ rules")
        st.markdown("**Mandatory:** `sensor_id`, `timestamp`, `speed` (record rejected / sensor excluded if absent). "
                    "**Optional:** weather and holiday (flagged, e.g. `weather_available=0`).")
        st.dataframe(load_csv("dq_rules.csv"), hide_index=True, width="stretch")
    else:
        missing_notice(rules_path, "Run `python -m src.benchmark.curate`.")
    if missing_path.exists():
        st.subheader("Missing / failed records by source")
        st.dataframe(load_csv("missing_by_source.csv"), hide_index=True, width="stretch")

    demo = BENCH / "fault_injection_demo.md"
    if demo.exists():
        with st.expander("Fault-injection demo: weather API outage + missing calendar file"):
            st.markdown(demo.read_text(encoding="utf-8"))

    path = BENCH / "dq_summary.md"
    if not path.exists():
        missing_notice(path, "Run `python -m src.benchmark.run`.")
        return
    md = path.read_text(encoding="utf-8")
    with st.expander("Full DQ summary"):
        st.markdown(md)
    rows = parse_dq_rows(md)
    if rows.empty:
        return
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(rows, x="pct", y="item", orientation="h", title="Missing / flagged (% of readings)",
                           labels={"pct": "%", "item": ""}, height=380), width="stretch")
    flags = rows.dropna(subset=["count"])
    c2.plotly_chart(px.bar(flags, x="count", y="item", orientation="h", title="Flag counts",
                           labels={"count": "rows", "item": ""}, height=380), width="stretch")


tabs = st.tabs([
    "Overview", "Bengaluru Live Congestion", "Model Validation (METR-LA benchmark)",
    "Route Optimization (METR-LA benchmark)", "Data Quality (METR-LA benchmark)",
])
for tab, render in zip(tabs, (tab_overview, tab_live, tab_model, tab_routing, tab_dq)):
    with tab:
        render()
