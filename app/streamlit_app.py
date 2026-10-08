"""Streamlit dashboard. Reads only saved outputs (reports/benchmark/, data/raw/traffic_api/).

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(BENCH / "test_predictions.parquet")
    df["actual"] = df["y_true"].map(dict(enumerate(CLASSES)))
    df["predicted"] = df["y_pred"].map(dict(enumerate(CLASSES)))
    return df


@st.cache_data
def load_importance() -> pd.DataFrame | None:
    path = BENCH / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else None


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


def best_model_name(metrics: pd.DataFrame) -> str:
    """Same rule as the training run: best learned model by validation macro-F1."""
    val = metrics[(metrics["split"] == "val") & (metrics["model"] != "persistence")]
    return str(val.loc[val["macro_f1"].idxmax(), "model"])


def missing_notice(path: Path, hint: str) -> None:
    st.info(f"`{path.relative_to(ROOT)}` not found yet. {hint}")


# --- tabs -----------------------------------------------------------------------------------

def tab_overview() -> None:
    st.header("Congestion prediction & route optimization")
    st.markdown(
        "Predict congestion level (**Low / Moderate / Severe**, from `speed_ratio = current speed / free-flow speed`) "
        "for road segments **30 minutes ahead**, then recommend congestion-aware routes versus plain "
        "shortest-distance routes. The model is first benchmarked on the public **METR-LA** dataset; "
        "the same pipeline is being applied to **Bengaluru** (South-East corridor) using live TomTom data."
    )

    st.subheader("Data sources")
    st.dataframe(pd.DataFrame([
        ["METR-LA sensors", "HDF5 + CSV", "5-min speeds (mph), 207 loop detectors (40 used)", "data/raw/metr_la/"],
        ["Open-Meteo weather", "JSON", "Hourly temperature, precipitation, humidity (Los Angeles)", "data/raw/metr_weather/"],
        ["US holidays", "CSV (`holidays` lib)", "2012 federal holidays", "data/raw/metr_calendar/"],
        ["Bengaluru TomTom live collector", "JSON (gzip)", "Flow Segment API, 13 segments, every 15 min via GitHub Actions", "data/raw/traffic_api/"],
    ], columns=["Source", "Type", "Content", "Raw zone"]), hide_index=True, width="stretch")

    st.subheader("Pipeline")
    st.graphviz_chart(
        "digraph { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica]; "
        'S [label="Sources"]; R [label="Raw zones\\n(append-only)"]; '
        'C [label="Cleaning /\\nmissing-data flags"]; I [label="Integration\\n(segment, time)"]; '
        'F [label="Features"]; M [label="Model"]; O [label="Routing"]; '
        "S -> R -> C -> I -> F -> M -> O }"
    )

    st.subheader("Bengaluru live collection")
    n_files, latest, snap = load_live_snapshot()
    if snap is None:
        st.info("No snapshots collected yet in `data/raw/traffic_api/`.")
        return
    c1, c2 = st.columns(2)
    c1.metric("Raw snapshot files", n_files)
    c2.metric("Latest snapshot", latest.replace("run_", ""))
    st.dataframe(snap, hide_index=True, width="stretch")


def tab_map() -> None:
    st.header("Congestion map (METR-LA, test set)")
    path = BENCH / "test_predictions.parquet"
    if not path.exists():
        missing_notice(path, "Run `python -m src.benchmark.export_predictions`.")
        return
    preds = load_predictions()
    times = sorted(preds["time"].unique())
    i = st.slider("Forecast origin (test set)", 0, len(times) - 1, len(times) // 2)
    t = pd.Timestamp(times[i])
    st.caption(
        f"Features up to **{t:%Y-%m-%d %H:%M}** -> congestion predicted for **{t + pd.Timedelta(minutes=30):%H:%M}** "
        "(30 min ahead). 'actual' is what was observed then."
    )
    now = preds[preds["time"] == t]

    left, right = st.columns([3, 1])
    fig = px.scatter_map(
        now, lat="lat", lon="lon", color="predicted", color_discrete_map=COLORS,
        category_orders={"predicted": CLASSES}, hover_name="sensor_id", hover_data={"actual": True, "lat": False, "lon": False},
        zoom=9.5, height=520, map_style="open-street-map",
    )
    fig.update_traces(marker={"size": 14})
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend_title_text="Predicted")
    left.plotly_chart(fig, width="stretch")

    counts = pd.concat([
        now["predicted"].value_counts().reindex(CLASSES, fill_value=0).rename("Predicted"),
        now["actual"].value_counts().reindex(CLASSES, fill_value=0).rename("Actual"),
    ], axis=1).reset_index(names="level").melt("level", var_name="series", value_name="sensors")
    bar = px.bar(counts, x="level", y="sensors", color="series", barmode="group", height=300,
                 category_orders={"level": CLASSES}, color_discrete_sequence=["#4c78a8", "#9aa5b1"])
    right.plotly_chart(bar, width="stretch")
    right.caption(f"{len(now)} sensors with a complete record at this time.")


def tab_model() -> None:
    st.header("Model results")
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
    st.caption("XGBoost (class-weighted) is highlighted. Severe recall matters most.")

    best = best_model_name(metrics)
    c1, c2 = st.columns(2)
    cm = BENCH / "confusion_matrix.png"
    if cm.exists():
        c1.image(str(cm), caption=f"Confusion matrix - {best}, test set")
    else:
        c1.info("confusion_matrix.png not found.")
    imp = load_importance()
    if imp is not None:
        top = imp.head(8).iloc[::-1]
        fig = px.bar(top, x="importance", y="feature", orientation="h", height=380,
                     title=f"Feature importance ({best}, impurity-based)")
        c2.plotly_chart(fig, width="stretch")

    st.subheader("SHAP")
    s1, s2 = st.columns(2)
    for col, name in ((s1, "shap_bar.png"), (s2, "shap_summary.png")):
        if (BENCH / name).exists():
            col.image(str(BENCH / name), caption=name)
        else:
            col.info(f"`reports/benchmark/{name}` not generated yet (SHAP step not run).")

    if imp is not None:
        names = imp["feature"].tolist()
        ratio_share = imp.loc[imp["feature"].str.startswith("ratio_"), "importance"].sum()
        st.markdown(
            f"- Top features by importance ({best}): **{names[0]}**, **{names[1]}**, **{names[2]}** "
            f"({imp['importance'].iloc[:3].sum():.0%} combined).\n"
            f"- Speed-ratio features (current, lags, 1 h stats) carry {ratio_share:.0%} of importance; "
            f"time, weather, holiday and sensor features share the remaining {1 - ratio_share:.0%}."
        )


def tab_routing() -> None:
    st.header("Route optimization")
    results = BENCH / "routing_results.csv"
    example = BENCH / "routing_example.png"
    if not results.exists():
        missing_notice(results, "The routing step (`src/routing`) has not been run yet.")
    else:
        df = pd.read_csv(results)
        saving_cols = [c for c in df.columns if "saving" in c.lower()]
        if saving_cols:
            c1, c2 = st.columns(2)
            c1.metric(f"Mean {saving_cols[0]}", f"{df[saving_cols[0]].mean():.2f}")
            c2.metric(f"Median {saving_cols[0]}", f"{df[saving_cols[0]].median():.2f}")
        st.dataframe(df, hide_index=True, width="stretch")
        pair = st.selectbox("O-D pair", df.index, format_func=lambda i: " -> ".join(str(v) for v in df.iloc[i, :2]))
        st.write(df.iloc[pair].to_frame("value"))
    if example.exists():
        st.image(str(example), caption="Example: shortest-distance vs congestion-aware route")
    else:
        missing_notice(example, "")


def tab_dq() -> None:
    st.header("Data quality")
    path = BENCH / "dq_summary.md"
    if not path.exists():
        missing_notice(path, "Run `python -m src.benchmark.run`.")
        return
    md = path.read_text(encoding="utf-8")
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


tabs = st.tabs(["Overview", "Congestion Map", "Model Results", "Route Optimization", "Data Quality"])
for tab, render in zip(tabs, (tab_overview, tab_map, tab_model, tab_routing, tab_dq)):
    with tab:
        render()
