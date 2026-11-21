"""Launder Lens 2.0 - Streamlit investigation platform.   Run:  python -m streamlit run app/app.py"""
import sys, time, json, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import REPORTS_DIR, MODELS_DIR, ARTIFACTS_DIR, TYPOLOGIES, TYPOLOGY_LABELS
from src.simulator import simulate, SimConfig
from src.detect import analyze, compare, typology_recovery
from src.evidence import find_rings, explain_account, case_narrative, ring_truth_overlap
import ui_kit as ui
from ui_kit import C, label, color
from graph_views import detection_graph, ring_graph, STATUS

st.set_page_config(page_title="Launder Lens 2.0", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

SCEN_LABEL = {"naive": "Naive launderer", "camouflaged": "Camouflaged amounts", "patient": "Camouflaged + patient"}
NEED = "Results not found. Run `python -m src.run_pipeline` once (about 10 minutes), then refresh this page."


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner=False)
def csv(name):
    p = REPORTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner=False)
def bundle():
    p = MODELS_DIR / "lab_models.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_resource(show_spinner=False)
def scenario(name):
    p = ARTIFACTS_DIR / f"scenario_{name}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


@st.cache_resource(show_spinner="Simulating a bank network and hunting for laundering rings ...", max_entries=6)
def run_lab(n, rings, typs, c, p, seed):
    sim = simulate(SimConfig(n_accounts=n, rings_per_type=rings, typologies=typs, camouflage=c, patience=p, seed=seed))
    return sim, analyze(sim, bundle())


@st.cache_resource(show_spinner="Analysing showcase network ...", max_entries=3)
def run_showcase(name):
    sc = scenario(name)
    return sc["sim"], analyze(sc["sim"], bundle())


@st.cache_data(show_spinner=False)
def rob_means():
    r = csv("robustness_grid.csv")
    return None if r is None else r.groupby(["model", "camouflage", "patience"])[["ap", "f1", "precision", "recall", "ring_recall"]].mean().reset_index()


def need_or_stop(*objs):
    if any(o is None for o in objs):
        st.warning(NEED); st.stop()


# ------------------------------------------------------------------ 1. Mission control
def page_home():
    c1, c2 = st.columns([1.25, 1], gap="medium")
    with c1:
        st.markdown("""<div class="hero"><div class="kicker"><span class="live"></span>Launder Lens 2.0 &middot; financial crime lab</div>
<h1>We built a fraud detector. <em>Then we tried to launder past it.</em></h1>
<p>Money laundering hides in networks, not in single transactions. Launder Lens simulates laundering rings, hunts them with graph machine learning,
attacks its own detectors &mdash; and shows exactly where they break and how to harden them.</p>
<div style="margin-top:1rem"><span class="chip red">Red team vs blue team</span><span class="chip teal">IBM AMLSim benchmark</span>
<span class="chip">5 laundering typologies</span><span class="chip amber">Unseen-pattern tests</span><span class="chip">Explainable cases</span></div></div>""", unsafe_allow_html=True)
    with c2:
        ui.hero_canvas(330)
    am, rb, lat = csv("amlsim_cv_summary.csv"), rob_means(), csv("ring_latency.csv")
    if am is None or rb is None:
        st.warning(NEED); return
    am = am.set_index("model")
    g = lambda m, c, p: float(rb[(rb.model == m) & (rb.camouflage == c) & (rb.patience == p)].ap.iloc[0])
    worst = float(rb[rb.model == "xgb_structure_hardened"].ap.min())
    sel = lat[(lat.scenario == "camouflaged") & (lat.model == "struct")] if lat is not None else None
    med = f"{sel.latency.dropna().median():.0f} steps" if sel is not None and sel.latency.notna().any() else "n/a"
    ui.kpis([("AMLSim benchmark - F1", f"{am.loc['xgb_all', 'f1_mean']:.2f}", "ring-grouped 5-fold CV, 10k accounts, 1.3M transfers", C["teal"]),
             ("Amount-only detector, AP", f"{g('xgb_amount', 0, 0):.2f} → {g('xgb_amount', 1, 0):.2f}", "clean laundering vs disguised amounts", C["signal"]),
             ("Hardened structure-first, worst AP", f"{worst:.2f}", "across all 15 attack settings (camouflage × patience)", C["steel"]),
             ("Median time to detect a ring", med, "from first laundering transfer to alert (camouflaged)", C["amber"])])
    st.markdown('<div class="flow"><span class="st">Simulate</span><span class="ar">→</span><span class="st">Analyse</span><span class="ar">→</span>'
                '<span class="st">Detect</span><span class="ar">→</span><span class="st">Trace</span><span class="ar">→</span><span class="st">Explain</span>'
                '<span class="ar">→</span><span class="st">Investigate</span><span class="ar">→</span><span class="st">Adapt</span></div>', unsafe_allow_html=True)
    sh = csv("amlsim_shap.csv"); share = sh.groupby("group").mean_abs_shap.sum() / sh.mean_abs_shap.sum()
    ls = csv("loto_results.csv"); l1 = ls[ls.camouflage == 1.0].groupby("model").recall.mean()
    acts = [("01", "Benchmark", f"On real-format AMLSim data a hybrid model reaches F1 {am.loc['xgb_all', 'f1_mean']:.2f} - but {share.get('amount', 0):.0%} of its SHAP weight sits on transaction amounts."),
            ("02", "Expose", f"Disguise the amounts and the amount-only model drops from AP {g('xgb_amount', 0, 0):.2f} to {g('xgb_amount', 1, 0):.2f}. The accuracy was partly a simulator shortcut."),
            ("03", "Defend", f"Structure-first models ignore amounts. Adversarial hardening keeps AP at {worst:.2f} or better even against patient, camouflaged launderers."),
            ("04", "Investigate", f"Flagged accounts are grouped into rings, traced through time and explained with exact TreeSHAP reasons an investigator can read.")]
    cols = st.columns(4)
    for col, (n, t, b) in zip(cols, acts):
        col.markdown(f'<div class="card"><div class="act"><div class="n">{n}</div><div><h4>{t}</h4><p>{b}</p></div></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure()
    for m in ["rules", "xgb_amount", "xgb_all", "xgb_structure", "xgb_structure_hardened", "gnn_structure_hardened"]:
        d = rb[(rb.model == m) & (rb.patience == 0)].sort_values("camouflage")
        fig.add_trace(go.Scatter(x=d.camouflage, y=d.ap, name=label(m), mode="lines+markers",
                                 line=dict(color=color(m), width=3 if "hardened" in m else 2, dash="solid" if "hardened" in m or m in ("xgb_amount", "xgb_all") else "dot"),
                                 marker=dict(size=8)))
    fig.update_layout(title="Disguise the amounts, and the detectors that relied on them collapse",
                      xaxis_title="Amount camouflage (0 = naive launderer, 1 = amounts indistinguishable from normal traffic)", yaxis_title="Average precision (PR-AUC)",
                      yaxis_range=[0, 1.02])
    ui.plot(fig, 420, key="home_curve")
    ui.insight("<b>Why this matters.</b> A detector that scores 0.9 on a benchmark can be defeated by a launderer who simply changes how much money moves. "
               "Structure and timing are far harder to fake - and that is what graph machine learning can exploit.")


# ------------------------------------------------------------------ 2. Simulation lab
def page_lab():
    ui.page_head("Simulate → Detect", "Simulation Lab",
                 "Generate a synthetic bank with hidden laundering rings, choose how cleverly the launderers behave, and watch three detectors try to find them.")
    need_or_stop(bundle())
    ss = st.session_state
    ss.setdefault("lab_params", dict(n=2000, rings=6, typs=tuple(TYPOLOGIES), c=1.0, p=0.0, seed=7))
    P = ss.lab_params
    k1, k2 = st.columns([1, 1.2], gap="large")
    with k1:
        attack = st.radio("Launderer behaviour", ["Naive launderer", "Camouflaged amounts", "Camouflaged + patient", "Custom"], horizontal=False,
                          index=["Naive launderer", "Camouflaged amounts", "Camouflaged + patient", "Custom"].index(ss.get("attack", "Camouflaged amounts")), key="attack")
        c, p = {"Naive launderer": (0.0, 0.0), "Camouflaged amounts": (1.0, 0.0), "Camouflaged + patient": (1.0, 0.5)}.get(attack, (P["c"], P["p"]))
        if attack == "Custom":
            c = st.slider("Amount camouflage", 0.0, 1.0, float(P["c"]), 0.25)
            p = st.slider("Patience (slower laundering)", 0.0, 1.0, float(P["p"]), 0.25)
    with k2:
        n = st.select_slider("Accounts", [1000, 1500, 2000, 3000, 4000], P["n"])
        rings = st.slider("Rings per typology", 2, 12, P["rings"])
        typs = st.multiselect("Laundering typologies", TYPOLOGIES, list(P["typs"]), format_func=lambda t: TYPOLOGY_LABELS[t])
        seed = st.number_input("Random seed", 0, 9999, P["seed"])
        go_ = st.button("Generate network & run detectors", type="primary")
    if go_:
        ss.lab_params = dict(n=n, rings=rings, typs=tuple(typs or TYPOLOGIES), c=float(c), p=float(p), seed=int(seed))
    P = ss.lab_params
    sim, an = run_lab(P["n"], P["rings"], P["typs"], P["c"], P["p"], P["seed"])
    ss.lab = dict(sim=sim, an=an)
    y = sim.accounts.is_fraud.values
    ui.kpis([("Transfers", f"{len(sim.tx):,}", f"{P['n']:,} accounts over 200 steps", C["steel"]),
             ("Hidden mule accounts", f"{int(y.sum())}", f"{y.mean():.1%} of accounts", C["signal"]),
             ("Laundering rings", f"{len(sim.rings)}", ", ".join(sorted({TYPOLOGY_LABELS[t].split(' (')[0] for t in sim.rings.typology})), C["amber"]),
             ("Camouflage / patience", f"{P['c']:.2f} / {P['p']:.2f}", "red-team settings used for this network", C["violet"])])
    cmp_ = compare(sim, an).set_index("model")
    cols = st.columns(3)
    for col, m in zip(cols, ["hybrid_naive", "hybrid_hardened", "structure_hardened"]):
        r = cmp_.loc[m]
        col.markdown(f'<div class="kpi" style="--accent:{color(m)}"><div class="l">{label(m)}</div><div class="v">F1 {r.f1:.2f}</div>'
                     f'<div class="s">recall {r.recall:.0%} · precision {r.precision:.0%}<br>rings recovered {r.ring_recall:.0%} · AP {r.ap:.2f}</div></div>', unsafe_allow_html=True)
    fig = go.Figure()
    for met, nm in [("f1", "F1"), ("recall", "Recall"), ("precision", "Precision")]:
        fig.add_trace(go.Bar(x=[label(m) for m in cmp_.index], y=cmp_[met], name=nm))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05], title="Same network, three detectors", colorway=[C["steel"], C["teal"], C["amber"]])
    ui.plot(fig, 330, key="lab_bars")
    view = st.radio("Show detections from", list(an["flags"]), format_func=label, horizontal=True, index=2)
    nodes, edges, counts = detection_graph(sim.tx, y.astype(bool), an["flags"][view])
    ui.legend([(f"caught mule ({counts['tp']})", STATUS["tp"][0]), (f"missed mule ({counts['fn']})", STATUS["fn"][0]),
               (f"false alarm ({counts['fp']})", STATUS["fp"][0]), ("legitimate neighbour", "#38465a")])
    ui.network(nodes, edges, 600)
    rec = typology_recovery(sim, an["flags"][view]); rec["typology"] = rec.typology.map(TYPOLOGY_LABELS)
    rec["ring_recall"] = (rec.ring_recall * 100).round(0).astype(int).astype(str) + "%"
    rec["member_coverage"] = (rec.member_coverage * 100).round(0).astype(int).astype(str) + "%"
    st.markdown("##### Ring recovery by typology")
    ui.table(rec.rename(columns={"typology": "Typology", "rings": "Rings", "rings_found": "Recovered (≥50% of members)", "ring_recall": "Ring recall", "member_coverage": "Member coverage"}))
    ui.insight("Try <b>Naive launderer</b> then <b>Camouflaged amounts</b>: the hybrid model trained on naive data falls apart, while the structure-first model barely moves. "
               "Then try <b>Camouflaged + patient</b> to see what a smarter attacker costs the defender.")


# ------------------------------------------------------------------ 3. Time machine
def page_time():
    ui.page_head("Detect over time", "Time Machine",
                 "Replay 200 simulation steps. Detectors re-score the network at each snapshot using only the transfers that have happened so far.")
    avail = [s for s in SCEN_LABEL if scenario(s) is not None]
    need_or_stop(None if not avail else 1)
    a, b = st.columns([1.4, 1])
    scen = a.radio("Scenario", avail, format_func=SCEN_LABEL.get, horizontal=True)
    model = b.selectbox("Detector", ["struct", "gnn", "all"], format_func=label)
    sc = scenario(scen); sim, ts = sc["sim"], sc["snap_ts"]
    idx = st.select_slider("Simulation step", options=list(range(len(ts))), value=len(ts) // 2, format_func=lambda i: f"step {ts[i]}")
    play = st.button("▶  Replay the whole timeline")
    ph_k, ph_g = st.empty(), st.empty()

    def render(i):
        flags = sc["scores"][model][i] >= sc["thr"][model]; truth = sc["y_t"][i].astype(bool)
        nodes, edges, counts = detection_graph(sim.tx, truth, flags, ts_max=ts[i])
        with ph_k.container():
            ui.kpis([("Simulation step", f"{ts[i]}", f"snapshot {i + 1} of {len(ts)}", C["steel"]),
                     ("Mules active so far", f"{int(truth.sum())}", "accounts with ≥1 laundering transfer", C["signal"]),
                     ("Caught", f"{int((flags & truth).sum())}", f"of {int(truth.sum())} active mules", C["teal"]),
                     ("False alarms", f"{int((flags & ~truth).sum())}", "legitimate accounts flagged", C["violet"])])
        with ph_g.container():
            ui.network(nodes, edges, 560)
    if play:
        for i in range(len(ts)):
            render(i); time.sleep(0.8)
    else:
        render(idx)
    cur, lat = csv("replay_curves.csv"), csv("ring_latency.csv")
    if cur is None:
        return
    d = cur[(cur.scenario == scen) & (cur.model == model)]
    f = go.Figure()
    f.add_trace(go.Scatter(x=d.ts, y=d.fraud_so_far, name="Mules active", line=dict(color="#8a94a3", dash="dot", width=2)))
    f.add_trace(go.Scatter(x=d.ts, y=d.true_flagged, name="Caught", line=dict(color=C["teal"], width=3), fill="tozeroy", fillcolor="rgba(61,214,192,.12)"))
    f.add_trace(go.Scatter(x=d.ts, y=d.false_flagged, name="False alarms", line=dict(color=C["violet"], width=2)))
    f.update_layout(title=f"{SCEN_LABEL[scen]} · {label(model)}", xaxis_title="Simulation step", yaxis_title="Accounts")
    c1, c2 = st.columns(2)
    with c1:
        ui.plot(f, 340, key="tm_flow")
    d2 = cur[cur.scenario == scen].copy(); d2["recall"] = d2.true_flagged / d2.fraud_so_far.clip(lower=1)
    f2 = go.Figure()
    for m in ["struct", "gnn", "all"]:
        x = d2[d2.model == m]
        f2.add_trace(go.Scatter(x=x.ts, y=x.recall, name=label(m), line=dict(color=color(m), width=3)))
    f2.update_layout(title="Share of active mules caught - all detectors", yaxis_range=[0, 1.02], xaxis_title="Simulation step")
    with c2:
        ui.plot(f2, 340, key="tm_cmp")
    if lat is not None:
        l = lat[lat.scenario == scen]
        s = l.groupby(["model", "typology"]).agg(rings=("ring_id", "size"), detected=("detect_ts", lambda x: x.notna().mean()),
                                                 median_latency=("latency", "median")).reset_index()
        s["Typology"] = s.typology.map(TYPOLOGY_LABELS); s["Detector"] = s.model.map(label)
        s["Detected"] = (s.detected * 100).round(0).astype(int).astype(str) + "%"
        s["Median latency (steps)"] = s.median_latency.round(0)
        st.markdown("##### Detection latency per ring")
        ui.table(s[["Detector", "Typology", "rings", "Detected", "Median latency (steps)"]].rename(columns={"rings": "Rings"}))


# ------------------------------------------------------------------ 4. Case desk
def page_case():
    ui.page_head("Trace → Explain → Investigate", "Case Desk",
                 "Flagged accounts are grouped into rings. Open a case to see the money trail, the exact reasons behind each risk score, and write your findings.")
    need_or_stop(bundle())
    ss = st.session_state
    ss.setdefault("cases", [])
    a, b = st.columns([1.3, 1])
    src_opts = (["Lab network (from Simulation Lab)"] if "lab" in ss else []) + [f"Showcase: {SCEN_LABEL[s]}" for s in SCEN_LABEL if scenario(s) is not None]
    need_or_stop(None if not src_opts else 1)
    src = a.selectbox("Data source", src_opts)
    model = b.selectbox("Detector", list(bundle()["models"]), index=1, format_func=label)
    if src.startswith("Lab"):
        sim, an = ss.lab["sim"], ss.lab["an"]
    else:
        sim, an = run_showcase([k for k, v in SCEN_LABEL.items() if v in src][0])
    ids = sim.accounts.account_id.values
    rings = find_rings(sim.tx, an["flags"][model], ids, an["scores"][model])
    if not rings:
        st.info("No rings flagged by this detector on this network."); return
    pick = st.selectbox("Case", range(len(rings)), format_func=lambda i: f"{rings[i]['case_id']} · {rings[i]['shape']} · {len(rings[i]['members'])} accounts · risk {rings[i]['risk']:.2f}")
    ring = rings[pick]
    truth = ring_truth_overlap(ring, sim.accounts)
    scores = an["scores"][model]
    tm = bundle()["models"][model]
    top_acc = max(ring["members"], key=lambda m: scores[m])
    reasons = explain_account(tm, an["feats"], top_acc, 4)
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        nodes, edges = ring_graph(sim.tx, ring)
        hier = ring["shape"] in ("Fan-in (collector)", "Fan-out (dispersal)", "Layering (mule chain)", "Smurfing (split & merge)") and len(nodes) <= 40
        ui.legend([("entry point", C["teal"]), ("exit point", C["amber"]), ("intermediary", C["signal"])])
        ui.network(nodes, edges, 470, hierarchical=hier)
    with c2:
        st.markdown(f'<div class="case">{case_narrative(ring, reasons, truth).replace(chr(10), "<br>")}</div>'.replace("**", ""), unsafe_allow_html=False) if False else None
        st.markdown(f'<div class="case">', unsafe_allow_html=True)
        st.markdown(case_narrative(ring, reasons, truth))
        st.markdown("</div>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Risk drivers", "Money trace", "Accounts", "Case file"])
    with t1:
        acc = st.selectbox("Account", sorted(ring["members"], key=lambda m: -scores[m]), format_func=lambda m: f"account {m} · risk {scores[m]:.2f}")
        rs = explain_account(tm, an["feats"], acc, 8)
        fig = go.Figure(go.Bar(y=[r["label"] for r in rs][::-1], x=[r["shap"] for r in rs][::-1], orientation="h",
                               marker_color=[C["signal"] if r["shap"] > 0 else C["teal"] for r in rs][::-1],
                               hovertext=[r["text"] for r in rs][::-1]))
        fig.update_layout(title=f"Why account {acc} scores {scores[acc]:.2f} (TreeSHAP, log-odds)", xaxis_title="push towards fraud →")
        ui.plot(fig, 360, key="case_shap")
        for r in rs[:5]:
            st.markdown(f"- **{r['label']}** - this account {r['text']}.")
    with t2:
        e = sim.tx[sim.tx.src.isin(ring["members"]) & sim.tx.dst.isin(ring["members"])].sort_values("ts")
        order = {m: i for i, m in enumerate(dict.fromkeys(list(ring["entry"]) + list(e.src) + list(e.dst)))}
        xs, ys = [], []
        for r in e.itertuples():
            xs += [r.ts, r.ts, None]; ys += [order[r.src], order[r.dst], None]
        f = go.Figure()
        f.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(255,120,90,.7)", width=2), name="transfer", hoverinfo="skip"))
        f.add_trace(go.Scatter(x=e.ts, y=[order[m] for m in e.dst], mode="markers", marker=dict(size=np.clip(np.log1p(e.amount) * 2.4, 5, 22), color=C["signal"], line=dict(color="#fff", width=1)),
                               text=[f"{r.src} → {r.dst}: {r.amount:,.0f}" for r in e.itertuples()], hoverinfo="text", name="received"))
        f.update_layout(title="Money-trace ladder: each vertical line is one transfer", xaxis_title="Simulation step",
                        yaxis=dict(tickmode="array", tickvals=list(order.values()), ticktext=[f"acct {m}" for m in order], autorange="reversed"))
        ui.plot(f, 380, key="case_ladder")
        import networkx as nx
        g = e.groupby(["src", "dst"]).amount.sum().reset_index()
        H = nx.DiGraph(); H.add_edges_from(zip(g.src, g.dst))
        if nx.is_directed_acyclic_graph(H) and 2 <= len(H) <= 40:
            lab = list(H.nodes); ix = {n: i for i, n in enumerate(lab)}
            sk = go.Figure(go.Sankey(node=dict(label=[f"acct {n}" for n in lab], color=C["signal"], pad=14),
                                     link=dict(source=[ix[s] for s in g.src], target=[ix[d] for d in g.dst], value=g.amount.tolist(), color="rgba(255,120,90,.35)")))
            sk.update_layout(title="Where the money went")
            ui.plot(sk, 380, key="case_sankey")
        else:
            st.caption("This ring contains a cycle, so a one-way flow diagram does not apply - the ladder above shows the loop in time.")
    with t3:
        tbl = pd.DataFrame({"Account": ring["members"], "Risk score": [round(float(scores[m]), 3) for m in ring["members"]],
                            "Role": ["entry" if m in ring["entry"] else ("exit" if m in ring["exit"] else "intermediary") for m in ring["members"]],
                            "Transfers": [int(((sim.tx.src == m) | (sim.tx.dst == m)).sum()) for m in ring["members"]]})
        tbl["Ground truth"] = ["mule" if sim.accounts.is_fraud.iloc[m] else "legitimate" for m in ring["members"]]
        ui.table(tbl.sort_values("Risk score", ascending=False))
    with t4:
        status = st.selectbox("Status", ["Open", "Escalated to SAR", "Closed - false positive", "Needs more evidence"])
        notes = st.text_area("Analyst notes", placeholder="Observations, hypotheses, next actions ...", height=120)
        if st.button("Save to case register"):
            ss.cases = [x for x in ss.cases if x["case_id"] != ring["case_id"]] + [dict(
                case_id=ring["case_id"], shape=ring["shape"], accounts=ring["members"], risk=round(ring["risk"], 3), volume=round(ring["volume"], 2),
                status=status, notes=notes, narrative=case_narrative(ring, reasons), saved=time.strftime("%Y-%m-%d %H:%M"))]
            st.success(f"{ring['case_id']} saved.")
        if ss.cases:
            ui.table(pd.DataFrame(ss.cases)[["case_id", "shape", "risk", "volume", "status", "saved"]])
            st.download_button("Download case register (JSON)", json.dumps(ss.cases, indent=2), "launder_lens_cases.json", "application/json")


# ------------------------------------------------------------------ 5. Evidence lab
def page_evidence():
    ui.page_head("Experiments", "Evidence Lab", "Every number in this project comes from one of four experiments. Nothing here is hand-typed.")
    am, rb, lo, lat = csv("amlsim_cv_summary.csv"), rob_means(), csv("loto_results.csv"), csv("ring_latency.csv")
    need_or_stop(am, rb, lo)
    t1, t2, t3, t4 = st.tabs(["1 · AMLSim benchmark", "2 · Red team vs blue team", "3 · Unseen typologies", "4 · Detection latency"])
    with t1:
        order = ["rules", "xgb_amount", "xgb_structure", "gnn_structure", "rf_all", "gnn_all", "xgb_all"]
        a = am.set_index("model").loc[order].reset_index()
        f = go.Figure()
        f.add_trace(go.Bar(x=[label(m) for m in a.model], y=a.f1_mean, error_y=dict(type="data", array=a.f1_std), name="F1", marker_color=[color(m) for m in a.model]))
        f.add_trace(go.Scatter(x=[label(m) for m in a.model], y=a.ap_mean, name="PR-AUC", mode="markers", marker=dict(size=13, symbol="diamond", color="#fff", line=dict(color="#000", width=1))))
        f.update_layout(title="Ring-grouped 5-fold cross-validation (mean ± std)", yaxis_range=[0, 1.05])
        ui.plot(f, 380, key="am_bar")
        ui.insight("Grouped folds keep every laundering ring entirely inside train <i>or</i> test, so rings cannot be memorised. "
                   "Adding graph and timing features to amounts lifts F1 well beyond amounts alone - the graph does contribute on this benchmark.")
        c1, c2 = st.columns(2)
        pt = csv("amlsim_per_typology.csv")
        if pt is not None:
            ptp = pt.pivot(index="model", columns="typology", values="recall").loc[["rules", "xgb_amount", "xgb_structure", "xgb_all"]]
            f = go.Figure()
            for t, nm in [("cycle", "Cycle rings"), ("fan_in", "Fan-in rings")]:
                f.add_trace(go.Bar(x=[label(m) for m in ptp.index], y=ptp[t], name=nm))
            f.update_layout(barmode="group", title="Recall by laundering typology", yaxis_range=[0, 1.05], colorway=[C["signal"], C["teal"]])
            with c1:
                ui.plot(f, 340, key="am_typ")
        sh = csv("amlsim_shap.csv")
        top = sh.head(12).iloc[::-1]
        f = go.Figure(go.Bar(y=top.feature, x=top.mean_abs_shap, orientation="h", marker_color=[{"amount": C["amber"], "structure": C["teal"], "timing": C["steel"]}[g] for g in top.group]))
        share = sh.groupby("group").mean_abs_shap.sum() / sh.mean_abs_shap.sum()
        f.update_layout(title=f"Top drivers (amount {share['amount']:.0%} · structure {share['structure']:.0%} · timing {share['timing']:.0%})", xaxis_title="mean |SHAP|")
        with c2:
            ui.plot(f, 340, key="am_shap")
    with t2:
        models = ["xgb_amount", "xgb_all", "xgb_structure", "xgb_structure_hardened", "gnn_structure_hardened", "xgb_all_hardened"]
        fig = make_subplots(rows=2, cols=3, subplot_titles=[label(m) for m in models], horizontal_spacing=.06, vertical_spacing=.18)
        scale = [[0, "#1a0f12"], [.35, "#ff5a3c"], [.7, "#ffb347"], [1, "#3dd6c0"]]
        for k, m in enumerate(models):
            d = rb[rb.model == m].pivot(index="patience", columns="camouflage", values="ap")
            fig.add_trace(go.Heatmap(z=d.values, x=[f"{c:g}" for c in d.columns], y=[f"{p:g}" for p in d.index], zmin=0, zmax=1, colorscale=scale, showscale=k == 5,
                                     text=np.round(d.values, 2), texttemplate="%{text}", hovertemplate="camouflage %{x}<br>patience %{y}<br>AP %{z:.2f}<extra></extra>"),
                          row=k // 3 + 1, col=k % 3 + 1)
        fig.update_xaxes(title_text="camouflage", title_font_size=10); fig.update_yaxes(title_text="patience", title_font_size=10)
        fig.update_layout(title="Average precision under attack (rows: patience, columns: camouflage)")
        ui.plot(fig, 560, key="rb_heat")
        w = lambda m: rb[rb.model == m].ap.min()
        ui.insight(f"<b>Red team:</b> camouflage destroys amount features (amount-only worst AP {w('xgb_amount'):.2f}); patience degrades timing features. "
                   f"<b>Blue team:</b> training on attacked networks (hardening) lifts the structure-first model's worst case from {w('xgb_structure'):.2f} to {w('xgb_structure_hardened'):.2f}. "
                   f"Defence is an arms race, not a solved problem.")
        rr = rb[(rb.patience == 0)].pivot(index="camouflage", columns="model", values="ring_recall")[["xgb_amount", "xgb_all", "xgb_structure_hardened"]]
        f = go.Figure([go.Scatter(x=rr.index, y=rr[m], name=label(m), line=dict(color=color(m), width=3), mode="lines+markers") for m in rr.columns])
        f.update_layout(title="Rings recovered (≥50% of members flagged) vs camouflage", yaxis_range=[0, 1.02], xaxis_title="camouflage")
        ui.plot(f, 330, key="rb_ring")
    with t3:
        c = st.radio("Amounts in the simulated world", [1.0, 0.0], format_func=lambda v: "Camouflaged (no amount shortcut)" if v == 1.0 else "Naive (amounts give the game away)", horizontal=True)
        met = st.radio("Metric", ["ap", "f1", "recall"], format_func=lambda m: {"recall": "Recall", "ap": "PR-AUC", "f1": "F1"}[m], horizontal=True)
        if met == "recall":
            st.caption("Recall alone can be inflated by flagging almost everything (look at the rule baseline). PR-AUC is the fair default.")
        d = lo[lo.camouflage == c].groupby(["model", "held_out"])[met].mean().reset_index().pivot(index="model", columns="held_out", values=met)
        d = d.loc[["rules", "xgb_amount", "xgb_structure", "xgb_all", "gnn_structure"], TYPOLOGIES]
        f = go.Figure(go.Heatmap(z=d.values, x=[TYPOLOGY_LABELS[t].split(" (")[0] for t in d.columns], y=[label(m) for m in d.index], zmin=0, zmax=1,
                                 colorscale=[[0, "#1a0f12"], [.35, "#ff5a3c"], [.7, "#ffb347"], [1, "#3dd6c0"]], text=np.round(d.values, 2), texttemplate="%{text}"))
        f.update_layout(title="Trained WITHOUT the column's typology, tested ONLY on it (never seen)", xaxis_title="Held-out typology")
        ui.plot(f, 380, key="loto")
        ui.insight("This is the generalisation test: can a model recognise a laundering pattern it has never been shown? "
                   "Honest reading: structural signals transfer unevenly - dispersal and smurfing patterns generalise, circular patterns are hard to recognise if never seen. Unseen typologies remain an open problem.")
    with t4:
        need_or_stop(lat)
        s = lat.groupby(["scenario", "model"]).agg(detected=("detect_ts", lambda x: x.notna().mean()), median_latency=("latency", "median")).reset_index()
        s["scenario"] = s.scenario.map(SCEN_LABEL)
        f = go.Figure()
        for m in ["struct", "gnn", "all"]:
            x = s[s.model == m]
            f.add_trace(go.Bar(x=x.scenario, y=x.detected, name=label(m), marker_color=color(m), text=(x.median_latency.round(0).astype(str) + " steps"), textposition="outside"))
        f.update_layout(barmode="group", title="Share of rings detected (label: median steps from first laundering transfer to alert)", yaxis_range=[0, 1.15])
        ui.plot(f, 380, key="lat_bar")


# ------------------------------------------------------------------ router
PAGES = {"Mission Control": page_home, "Simulation Lab": page_lab, "Time Machine": page_time,
         "Case Desk": page_case, "Evidence Lab": page_evidence}
with st.sidebar:
    ui.brand()
    nav = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.markdown('<div class="side-note">IBM AMLSim · 10,000 accounts · 1.32M transfers<br>Simulator · 5 laundering typologies<br>XGBoost · Random Forest · SIGN-GNN · TreeSHAP</div>', unsafe_allow_html=True)
PAGES[nav]()
