"""Congestion-aware vs shortest-distance routing on the METR-LA sensor graph.

Graph: directed, sensors as nodes, edges from distances_la.csv (< 5 km, 40 sampled sensors,
largest weakly connected component). Edge time = distance / speed at the destination sensor.
Both routes are *evaluated* with the actual speeds 30 min after the forecast origin; the
congestion-aware route is *chosen* with XGBoost-predicted speeds only.

Run: ``python -m src.routing.benchmark_router``
"""
from __future__ import annotations

import json
import random
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.benchmark.prepare import (  # noqa: E402
    FEATURES, HORIZON, SEED, STEP_MIN, build_dataset, chronological_split, clean_speeds, load_raw_speeds, pick_sensors,
)
from src.utils.config import project_path  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

log = get_logger(__name__)
OUT = project_path("reports/benchmark")
DISTANCES = project_path("data/raw/metr_la/distances_la.csv")
LOCATIONS = project_path("data/raw/metr_la/sensor_locations_la.csv")
MAX_EDGE_M = 5000
N_PAIRS = 50
MPH_TO_MS = 0.44704
MIN_SPEED_MPH = 1.0


def travel_min(dist_m: float, speed_mph: float) -> float:
    return dist_m / (max(speed_mph, MIN_SPEED_MPH) * MPH_TO_MS) / 60


def build_graph(sensor_ids: list[str]) -> nx.DiGraph:
    """Directed sensor graph (edges < 5 km), restricted to the largest weakly connected component."""
    d = pd.read_csv(DISTANCES, dtype={"from": str, "to": str})
    keep = d["from"].isin(sensor_ids) & d["to"].isin(sensor_ids) & (d["from"] != d["to"]) & (d["cost"] < MAX_EDGE_M)
    g = nx.DiGraph()
    g.add_nodes_from(sensor_ids)
    g.add_weighted_edges_from(d.loc[keep, ["from", "to", "cost"]].itertuples(index=False), weight="dist")
    largest = max(nx.weakly_connected_components(g), key=len)
    return g.subgraph(largest).copy()


def pick_origin_time(test: pd.DataFrame, nodes: set[str], ids: np.ndarray, actual: pd.DataFrame) -> pd.Timestamp:
    """First weekday 08:00 test timestamp with complete features and actual speeds for every node."""
    cand = test[(test["time"].dt.weekday < 5) & (test["time"].dt.hour == 8) & (test["time"].dt.minute == 0)]
    for ts, rows in cand.groupby("time"):
        have = set(ids[rows["sensor"].to_numpy()])
        target = ts + pd.Timedelta(minutes=HORIZON * STEP_MIN)
        if nodes <= have and target in actual.index and actual.loc[target, list(nodes)].notna().all():
            return ts
    raise RuntimeError("no weekday 08:00 test timestamp with complete data")


def class_mean_ratio(train: pd.DataFrame) -> np.ndarray:
    """Mean speed ratio of each congestion class (0/1/2) in the training set."""
    return train.groupby(train["y_now"].astype(int))["ratio_t"].mean().reindex([0, 1, 2]).to_numpy()


def path_edges(path: list[str]) -> list[tuple[str, str]]:
    return list(zip(path[:-1], path[1:]))


def plot_example(g: nx.DiGraph, loc: pd.DataFrame, ratio_act: pd.Series, row: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for u, v in g.edges:
        ax.plot([loc.loc[u, "longitude"], loc.loc[v, "longitude"]], [loc.loc[u, "latitude"], loc.loc[v, "latitude"]],
                color="#cccccc", lw=0.8, zorder=1)
    colors = ["#2e9e4f" if r >= 0.75 else "#f28c28" if r >= 0.5 else "#d62728" for r in ratio_act[list(g.nodes)]]
    ax.scatter(loc.loc[list(g.nodes), "longitude"], loc.loc[list(g.nodes), "latitude"], c=colors, s=70, zorder=3,
               edgecolor="k", linewidth=0.5)
    for path, color, style, label in (
        (row["dist_path"], "#1f77b4", "--", f"shortest distance: {row['dist_route_min']:.1f} min"),
        (row["aware_path"], "#9467bd", "-", f"congestion-aware: {row['aware_route_min']:.1f} min"),
    ):
        xs, ys = loc.loc[path, "longitude"], loc.loc[path, "latitude"]
        ax.plot(xs, ys, color=color, ls=style, lw=3, zorder=2, label=label)
    for n, co in (("origin", row["origin"]), ("destination", row["destination"])):
        ax.annotate(n, (loc.loc[co, "longitude"], loc.loc[co, "latitude"]), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Largest saving of {N_PAIRS} O-D pairs ({row['saving_pct']:.1f}%) - evaluated with actual speeds\n"
                 "node colour = actual congestion (green Low, orange Moderate, red Severe)", fontsize=10)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    model = joblib.load(OUT / "models" / "best_model.pkl")["model"]
    df, stats = build_dataset()
    splits = chronological_split(df, stats["n_timestamps"])
    test, train = splits["test"], splits["train"]

    raw = pick_sensors(load_raw_speeds())
    ids = raw.columns.astype(str).to_numpy()
    actual = clean_speeds(raw)["filled"]
    actual.columns = ids

    g = build_graph(list(ids))
    nodes = set(g.nodes)
    log.info("graph: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())

    ts = pick_origin_time(test, nodes, ids, actual)
    target_ts = ts + pd.Timedelta(minutes=HORIZON * STEP_MIN)
    rows = test[test["time"] == ts]
    rows = rows[np.isin(ids[rows["sensor"].to_numpy()], list(nodes))]
    node_ids = ids[rows["sensor"].to_numpy()]
    expected_ratio = model.predict_proba(rows[FEATURES]) @ class_mean_ratio(train)
    pred_speed = pd.Series(expected_ratio * rows["free_flow_mph"].to_numpy(), index=node_ids)
    act_speed = actual.loc[target_ts, list(nodes)]
    free_flow = pd.Series(rows["free_flow_mph"].to_numpy(), index=node_ids)

    for u, v, data in g.edges(data=True):
        data["t_pred"] = travel_min(data["dist"], pred_speed[v])
        data["t_act"] = travel_min(data["dist"], act_speed[v])

    rng = random.Random(SEED)
    pairs = [(o, d) for o in g.nodes for d in g.nodes if o != d and nx.has_path(g, o, d)]
    sample = rng.sample(pairs, min(N_PAIRS, len(pairs)))

    def actual_time(path: list[str]) -> float:
        return sum(g[u][v]["t_act"] for u, v in path_edges(path))

    records = []
    for o, d in sample:
        p_dist = nx.shortest_path(g, o, d, weight="dist")
        p_aware = nx.shortest_path(g, o, d, weight="t_pred")
        t_dist, t_aware = actual_time(p_dist), actual_time(p_aware)
        records.append({
            "origin": o, "destination": d, "dist_route_min": round(t_dist, 3), "aware_route_min": round(t_aware, 3),
            "saving_pct": round(100 * (t_dist - t_aware) / t_dist, 2), "dist_path": p_dist, "aware_path": p_aware,
        })
    res = pd.DataFrame(records)
    res[["origin", "destination", "dist_route_min", "aware_route_min", "saving_pct"]].to_csv(
        OUT / "routing_results.csv", index=False
    )

    differ = res.apply(lambda r: r["dist_path"] != r["aware_path"], axis=1)
    meta = {
        "forecast_origin": str(ts), "evaluated_at": str(target_ts), "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(), "pairs": len(res), "pairs_with_different_routes": int(differ.sum()),
        "mean_saving_pct": round(float(res["saving_pct"].mean()), 2),
        "median_saving_pct": round(float(res["saving_pct"].median()), 2),
        "mean_saving_pct_differing_pairs": round(float(res.loc[differ, "saving_pct"].mean()), 2) if differ.any() else None,
        "pairs_improved": int((res["saving_pct"] > 0).sum()), "pairs_worse": int((res["saving_pct"] < 0).sum()),
    }
    (OUT / "routing_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    loc = pd.read_csv(LOCATIONS, dtype={"sensor_id": str}).set_index("sensor_id")
    best_row = (res[differ] if differ.any() else res).sort_values("saving_pct", ascending=False).iloc[0].to_dict()
    plot_example(g, loc, act_speed / free_flow, best_row, OUT / "routing_example.png")

    print(f"origin {ts} (evaluated at {target_ts}); {meta['nodes']} nodes, {meta['edges']} edges, {meta['pairs']} pairs")
    print(f"mean saving {meta['mean_saving_pct']:.2f}%  median saving {meta['median_saving_pct']:.2f}%  "
          f"(routes differ in {meta['pairs_with_different_routes']}/{meta['pairs']}; "
          f"improved {meta['pairs_improved']}, worse {meta['pairs_worse']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
