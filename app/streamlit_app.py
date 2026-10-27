"""ChurnLens Streamlit app — a thin UI over the same pipeline functions as the CLI.

Run:  streamlit run app/streamlit_app.py

Nothing uploaded is written to disk: uploads are processed in memory for the session only.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from churnlens import config  # noqa: E402

# Cap BLAS/OpenMP threads at the container's real CPU quota BEFORE numpy/sklearn/lightgbm load:
# cloud containers often report the host's core count, and oversubscribed threads crawl.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(config.n_jobs()))

from churnlens import io, models, run, segments, synthetic, threshold  # noqa: E402
from churnlens.io import ID, TARGET, Costs  # noqa: E402

st.set_page_config(page_title="ChurnLens", page_icon="📉", layout="wide")

SYNTHETIC = {
    "Synthetic telecom (labelled table)": (synthetic.labelled_config, synthetic.labelled),
    "Synthetic shop (raw transactions)": (synthetic.transactions_config, synthetic.transactions),
}


# ---------------------------------------------------------------- data sources
def real_demos() -> list[str]:
    """Bundled configs whose raw file is present locally (raw data is not in the repo)."""
    out = []
    for name in io.available_datasets():
        try:
            if io.load_config(name).path.exists():
                out.append(name)
        except io.ConfigError:
            continue
    return out


@st.cache_data(show_spinner="Reading raw file…")
def load_real(name: str) -> pd.DataFrame:
    return io.read_raw(io.load_config(name))


def upload_config(df: pd.DataFrame) -> io.DatasetConfig | None:
    """Column-mapping form for an uploaded CSV. Returns a config, or None until complete."""
    cols = list(df.columns)
    mode = st.sidebar.radio("What does one row represent?",
                            ["labelled", "transactions"],
                            format_func=lambda m: {"labelled": "One customer (has a churn column)",
                                                   "transactions": "One purchase"}[m])
    id_col = st.sidebar.selectbox("Customer ID column", cols)
    if mode == "labelled":
        label = st.sidebar.selectbox("Churn column", [c for c in cols if c != id_col])
        values = df[label].dropna().unique().tolist()[:50]
        positive = st.sidebar.selectbox("Value meaning 'churned'", values)
        columns = {"customer_id": id_col, "churn_label": label,
                   "churn_positive_value": positive.item() if hasattr(positive, "item")
                   else positive}
        extra = {}
    else:
        date = st.sidebar.selectbox("Date column", [c for c in cols if c != id_col])
        amount = st.sidebar.multiselect("Amount column(s) — several are multiplied",
                                        [c for c in cols if c not in (id_col, date)])
        invoice = st.sidebar.selectbox("Invoice / order column (optional)",
                                       ["(none)"] + [c for c in cols if c not in (id_col, date)])
        if not amount:
            st.sidebar.info("Pick at least one amount column.")
            return None
        columns = {"customer_id": id_col, "date": date,
                   "amount": amount if len(amount) > 1 else amount[0]}
        if invoice != "(none)":
            columns["invoice"] = invoice
        extra = {"transactions": {"inactivity_days": 90, "lookback_days": 365}}
    try:
        return io.parse_config({"name": "uploaded", "mode": mode, "path": "uploaded.csv",
                                "columns": columns, **extra})
    except io.ConfigError as e:
        st.sidebar.error(str(e))
        return None


def sidebar_source() -> tuple[str, io.DatasetConfig | None, pd.DataFrame | None]:
    st.sidebar.header("1 · Data")
    options = list(SYNTHETIC) + [f"Demo: {n}" for n in real_demos()] + ["Upload a CSV"]
    choice = st.sidebar.selectbox("Dataset", options)
    if choice in SYNTHETIC:
        make_cfg, make_data = SYNTHETIC[choice]
        return choice, make_cfg(), make_data()
    if choice.startswith("Demo: "):
        name = choice.removeprefix("Demo: ")
        return choice, io.load_config(name), load_real(name)
    file = st.sidebar.file_uploader("CSV file", type="csv")
    if file is None:
        return choice, None, None
    df = pd.read_csv(file)
    return f"upload:{file.name}:{len(df)}", upload_config(df), df


def sidebar_settings(cfg: io.DatasetConfig) -> io.DatasetConfig:
    st.sidebar.header("2 · Assumptions")
    st.sidebar.caption("Costs are **assumptions** — change them to your business's numbers.")
    c = cfg.costs
    costs = Costs(
        offer_cost=st.sidebar.number_input("Cost of one retention offer (₹)", 0.0,
                                           value=float(c.offer_cost), step=50.0),
        customer_value=st.sidebar.number_input("Value of keeping a customer (₹)", 0.0,
                                               value=float(c.customer_value), step=500.0),
        offer_success_rate=st.sidebar.slider("Share of targeted churners the offer saves",
                                             0.0, 1.0, float(c.offer_success_rate), 0.05),
    )
    tx = cfg.transactions
    if cfg.mode == "transactions":
        n = st.sidebar.select_slider("Inactivity window N (days without purchase = churned)",
                                     [30, 60, 90, 120, 180], value=cfg.inactivity_days)
        tx = {**tx, "inactivity_days": n}
    return dataclasses.replace(cfg, costs=costs, transactions=tx)


# ---------------------------------------------------------------- pipeline steps
def do_prepare(key: str, cfg: io.DatasetConfig, raw: pd.DataFrame) -> None:
    try:
        with st.spinner("Cleaning, splitting, checking for leakage…"):
            prep = run.prepare(cfg, raw=raw, save_figures=False)
    except (io.MappingError, io.ConfigError, ValueError) as e:
        st.error(f"Could not prepare the data: {e}")
        return
    st.session_state.update(key=key, prep=prep, trained=None)


def do_train(drop: list[str]) -> None:
    prep = st.session_state["prep"]
    with st.spinner("Cross-validating both models, choosing a threshold, computing SHAP…"):
        trained = run.train_and_evaluate(prep, drop=drop)
        decision = run.decide_threshold(prep, trained, save_figure=False)
        explained = run.explain_and_segment(prep, trained, decision, save_figure=False)
    st.session_state.update(trained=trained, decision=decision, explained=explained, drop=drop)


# ---------------------------------------------------------------- tabs
def tab_overview(prep: run.Prepared) -> None:
    s = prep.split_info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train rows", f"{s.get('train_rows', s.get('train_customers')):,}")
    c2.metric("Test rows", f"{s.get('test_rows', s.get('test_customers')):,}")
    c3.metric("Train churn rate", f"{s['train_churn_rate']:.1%}")
    c4.metric("Test churn rate", f"{s['test_churn_rate']:.1%}")
    st.write(f"**Split:** {s['method']}")
    if prep.cfg.mode == "transactions":
        st.info(
            f"Churn was **defined from inactivity**: a customer active in the "
            f"{s['lookback_days']} days before a cutoff is *churned* if they buy nothing in the "
            f"next {s['inactivity_days']} days. Train cutoff **{s['train_cutoff']}**, "
            f"test cutoff **{s['test_cutoff']}** — the model is judged on the future."
        )
    with st.expander("Cleaning log"):
        st.json(prep.cleaning_log)


def tab_leakage(prep: run.Prepared) -> list[str]:
    st.subheader("Leakage guard (training split only)")
    flags = pd.DataFrame([dataclasses.asdict(f) for f in prep.leakage.flags])
    if flags.empty:
        st.success("No columns flagged.")
    else:
        st.warning(f"{len(flags)} warning(s). Review them before training.")
        st.dataframe(flags, width="stretch", hide_index=True)
    auc = pd.Series(prep.leakage.single_feature_auc, name="single-feature AUC")
    st.caption("How well each column alone separates churners (0.5 = useless, > 0.95 = suspicious)")
    st.bar_chart(auc)
    feats = [c for c in prep.types if c not in (ID, TARGET)]
    flagged = [c for c in prep.leakage.flagged_columns if c in feats]
    default = sorted(set(prep.cfg.drop) | set(flagged))
    return st.multiselect("Columns to drop before training (you decide)", feats, default=default)


def tab_eda(prep: run.Prepared) -> None:
    e = prep.eda
    st.metric("Churn rate (train)", f"{e.churn_rate:.1%}")
    st.subheader("Findings")
    for f in e.findings:
        st.markdown(f"- {f}")
    if not e.groups.empty:
        col = st.selectbox("Churn rate by", list(dict.fromkeys(e.groups["column"])))
        g = e.groups[e.groups["column"] == col].sort_values("churn_rate")
        fig = px.bar(g, x="churn_rate", y="value", orientation="h", text_auto=".1%",
                     labels={"churn_rate": "churn rate", "value": col})
        fig.add_vline(x=e.churn_rate, line_dash="dash", annotation_text="overall")
        fig.update_layout(height=320, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    if not e.numeric.empty:
        st.subheader("Numeric differences (churned vs stayed)")
        st.dataframe(e.numeric, width="stretch", hide_index=True)


def tab_models(trained: run.Trained, decision: run.Decision) -> None:
    st.subheader("5-fold cross-validation on train")
    st.dataframe(trained.cv_table, width="stretch", hide_index=True)
    st.write(f"**Selected:** `{trained.best}` (highest CV PR-AUC — picked without the test set).")
    st.subheader("Test set at threshold 0.5")
    rows = [{"model": m, **{k: v for k, v in r.items() if k != "confusion_matrix"}}
            for m, r in trained.test_metrics.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    cm = decision.test_metrics_at_threshold["confusion_matrix"]
    st.subheader(f"Confusion matrix at the chosen threshold {decision.threshold:.2f}")
    st.table(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                          index=["actually stayed", "actually churned"],
                          columns=["predicted stay", "predicted churn"]))


def tab_threshold(prep: run.Prepared, trained: run.Trained, costs: Costs) -> float:
    y_test = prep.test[TARGET].to_numpy()
    oof_curve = threshold.savings_curve(prep.train[TARGET].to_numpy(),
                                        trained.oof[trained.best], costs)
    recommended = threshold.choose_threshold(oof_curve)
    t = st.slider("Threshold — target customers scoring at or above it", 0.05, 0.95,
                  recommended, 0.01)
    st.caption(f"Recommended for these costs: **{recommended:.2f}** (max savings on out-of-fold "
               "training predictions). Numbers below are on the test set. Costs are ASSUMED.")
    m = models.evaluate(y_test, trained.test_proba, t)
    strat = threshold.compare_strategies(y_test, trained.test_proba, costs, t)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected savings", f"₹{strat['chosen_threshold']['savings']:,.0f}")
    c2.metric("Customers targeted", f"{m['share_targeted']:.1%}")
    c3.metric("Recall (churners caught)", f"{m['recall']:.1%}")
    c4.metric("Precision", f"{m['precision']:.1%}")
    curve = threshold.savings_curve(y_test, trained.test_proba, costs)
    fig = go.Figure(go.Scatter(x=curve["threshold"], y=curve["savings"], mode="lines",
                               name="savings"))
    fig.add_vline(x=t, line_color="#E45756", annotation_text="selected")
    fig.add_vline(x=0.5, line_dash="dot", annotation_text="0.5")
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(height=340, xaxis_title="threshold", yaxis_title="expected savings (₹)")
    st.plotly_chart(fig, width="stretch")
    table = pd.DataFrame(strat).T.reset_index(names="strategy")
    table.loc[table["strategy"] == "chosen_threshold", "strategy"] = "selected threshold"
    st.dataframe(table, width="stretch", hide_index=True)
    return t


def tab_explain(prep: run.Prepared, explained: run.Explained) -> None:
    imp = explained.importance.head(12)[::-1]
    st.subheader("What the model relies on (mean |SHAP|, test customers)")
    st.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h",
                           labels={"x": "mean |SHAP| (log-odds)", "y": ""}).update_layout(
        height=360), width="stretch")
    st.subheader("One customer")
    cust = explained.customers.reset_index(drop=True)
    order = cust.sort_values("churn_score", ascending=False)
    pick = st.selectbox("Customer (highest risk first)", order[ID].astype(str).tolist()[:500])
    i = cust.index[cust[ID].astype(str) == pick][0]
    row = cust.loc[i]
    st.metric("Churn score", f"{row['churn_score']:.2f}")
    reasons = [row[f"reason_{k}"] for k in (1, 2, 3) if row[f"reason_{k}"]]
    st.markdown("**Top reasons for risk:**" if reasons else "No feature raises this risk.")
    for r in reasons:
        st.markdown(f"- {r}")
    contrib = explained.contributions.iloc[i].sort_values(key=abs, ascending=False).head(8)[::-1]
    fig = px.bar(x=contrib.values, y=contrib.index, orientation="h",
                 color=contrib.values > 0, color_discrete_map={True: "#E45756", False: "#4C78A8"},
                 labels={"x": "SHAP (→ raises risk, ← lowers)", "y": "", "color": "raises"})
    st.plotly_chart(fig.update_layout(height=320, showlegend=False), width="stretch")
    st.caption("SHAP shows what the model uses, not proven causes.")


def tab_segments(prep: run.Prepared, explained: run.Explained, t: float) -> None:
    seg = segments.assign_segments(explained.customers, t, prep.cfg.segments)
    st.write(f"**{len(seg):,}** test customers at or above threshold {t:.2f}, grouped by their "
             "main risk driver:")
    st.dataframe(segments.segment_summary(seg), width="stretch", hide_index=True)
    st.download_button("Download target list (IDs, segment, action)",
                       segments.targets_csv(seg).to_csv(index=False).encode(),
                       file_name=f"{prep.cfg.name}_targets.csv", mime="text/csv")
    st.caption("Actions are illustrative business rules from the dataset config.")


# ---------------------------------------------------------------- page
def main() -> None:
    st.title("ChurnLens")
    st.caption("Who is likely to leave, why, and who is worth an offer — for any customer data.")
    key, cfg, raw = sidebar_source()
    if cfg is None or raw is None:
        st.info("Upload a CSV and map its columns in the sidebar to begin.")
        return
    cfg = sidebar_settings(cfg)
    run_key = f"{key}|{cfg.inactivity_days}"
    if st.sidebar.button("Load & check data", type="primary") or (
            st.session_state.get("key") != run_key and "prep" not in st.session_state):
        do_prepare(run_key, cfg, raw)
    if st.session_state.get("key") != run_key:
        st.info("Dataset or window changed.")
        # A main-area button too: the sidebar one can be below the fold on small screens.
        if st.button("Load & check data", type="primary", key="load_main"):
            do_prepare(run_key, cfg, raw)
            st.rerun()
        return
    prep: run.Prepared = st.session_state["prep"]

    tabs = st.tabs(["Overview", "Leakage", "EDA", "Models", "Threshold", "Explain", "Segments"])
    with tabs[0]:
        tab_overview(prep)
    with tabs[1]:
        drop = tab_leakage(prep)
        if st.button("Train models", type="primary"):
            do_train(drop)
    with tabs[2]:
        tab_eda(prep)
    trained = st.session_state.get("trained")
    if trained is None:
        for tab in tabs[3:]:
            with tab:
                st.info("Review the Leakage tab, then press **Train models**.")
        return
    with tabs[3]:
        tab_models(trained, st.session_state["decision"])
    with tabs[4]:
        t = tab_threshold(prep, trained, cfg.costs)
    with tabs[5]:
        tab_explain(prep, st.session_state["explained"])
    with tabs[6]:
        tab_segments(prep, st.session_state["explained"], t)


main()
