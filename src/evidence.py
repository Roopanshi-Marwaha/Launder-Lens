"""Turns raw detections into investigator-ready evidence: ring extraction, money-flow tracing,
per-account TreeSHAP reasons, and a TEMPLATE-based case narrative (deliberately not an LLM, so the
report can never state a fact that is not in the data)."""
import numpy as np
import pandas as pd
import networkx as nx

FEATURE_TEXT = {
    "tmp_joined_in_burst": ("Part of a convergence", "sent money in the same step as {v:.0f} distinct accounts paying one receiver"),
    "tmp_joined_out_burst": ("Part of a dispersal", "received money in the same step that one sender paid {v:.0f} accounts"),
    "tmp_max_in_burst": ("Collector behaviour", "received from up to {v:.0f} distinct accounts in a single step"),
    "tmp_max_out_burst": ("Dispersal behaviour", "paid up to {v:.0f} distinct accounts in a single step"),
    "tmp_n_in_bursts": ("Repeated convergence", "had {v:.0f} step(s) with 4+ distinct senders"),
    "tmp_n_out_bursts": ("Repeated dispersal", "had {v:.0f} step(s) paying 4+ distinct receivers"),
    "tmp_fwd_fast_frac": ("Rapid forwarding", "forwarded {v:.0%} of incoming money within 2 steps"),
    "tmp_fwd_median_lag": ("Pass-through delay", "median delay between receiving and sending is {v:.0f} step(s)"),
    "tmp_out_tx": ("Outgoing activity", "made {v:.0f} outgoing transfers"),
    "tmp_in_tx": ("Incoming activity", "received {v:.0f} incoming transfers"),
    "tmp_velocity": ("Transaction velocity", "moves {v:.2f} transfers per active step"),
    "tmp_burstiness": ("Bursty activity", "{v:.0%} of its activity falls into its busiest step"),
    "tmp_active_steps": ("Active days", "was active on {v:.0f} distinct steps"),
    "tmp_span": ("Activity span", "was active over a span of {v:.0f} steps"),
    "str_wcycle_6": ("Fast closed loop", "sits on {v:.0f} short loop(s) that closed within 6 steps"),
    "str_wcycle_24": ("Closed loop", "sits on {v:.0f} loop(s) that closed within 24 steps"),
    "str_c2": ("Round-tripping", "has {v:.0f} reciprocal counterpart(s)"),
    "str_c3": ("Triangle loop", "lies on {v:.0f} directed 3-cycle(s)"),
    "str_c4": ("Square loop", "lies on {v:.0f} directed 4-cycle(s)"),
    "str_in_cycle": ("Cycle membership", "belongs to a short directed cycle"),
    "str_in_deg": ("Many payers", "receives from {v:.0f} distinct accounts"),
    "str_out_deg": ("Many payees", "pays {v:.0f} distinct accounts"),
    "str_deg_ratio": ("In/out imbalance", "in/out degree log-ratio is {v:.2f}"),
    "str_clustering": ("Tight neighbourhood", "clustering coefficient {v:.2f}"),
    "str_kcore": ("Dense core", "sits in the {v:.0f}-core of the network"),
    "str_pagerank": ("Money-flow importance", "PageRank score {v:.2f}x average"),
    "str_comm_size": ("Community size", "belongs to a community of {v:.0f} accounts"),
    "str_comm_density": ("Community density", "community density {v:.3f}"),
    "str_comm_internal_frac": ("Insular community", "{v:.0%} of its traffic stays inside its community"),
    "str_reciprocity": ("Reciprocity", "{v:.0%} of payees pay it back"),
    "str_nbr_mean_deg": ("Neighbour activity", "its counterparties average {v:.1f} connections"),
}


def _txt(feat, v):
    if feat.startswith("amt_"):
        return ("Amount pattern", f"{feat[4:].replace('_', ' ')} = {v:,.2f}")
    lab, tpl = FEATURE_TEXT.get(feat, (feat, "{v}"))
    return lab, tpl.format(v=v)


def explain_account(model, feats, account_id, top=5):
    row = feats.loc[[account_id]]
    c = model.contribs(row).iloc[0].drop("bias")
    out = []
    for f in c.abs().sort_values(ascending=False).index[:top]:
        lab, text = _txt(f, float(row[f].iloc[0]))
        out.append({"feature": f, "label": lab, "text": text, "shap": float(c[f]), "value": float(row[f].iloc[0])})
    return out


def shape_guess(H):
    if H.number_of_nodes() < 3:
        return "Too small to classify"
    ins = dict(H.in_degree()); outs = dict(H.out_degree())
    mi, mo = max(ins.values()), max(outs.values())
    if not nx.is_directed_acyclic_graph(H):
        return "Circular transfers"
    if mi >= 4 and mo >= 4 or (mi >= 4 and any(ins[n] >= 3 and outs[n] >= 1 for n in H) and mo >= 3):
        return "Smurfing (split & merge)"
    if mi >= 4:
        return "Fan-in (collector)"
    if mo >= 4:
        return "Fan-out (dispersal)"
    if mi <= 2 and mo <= 2:
        return "Layering (mule chain)"
    return "Mixed network"


def find_rings(tx, flags, ids, scores=None, min_size=3):
    """Weakly connected components of flagged accounts joined by transfers between them."""
    ids = np.asarray(ids)
    flagged = set(ids[flags])
    sub = tx[tx.src.isin(flagged) & tx.dst.isin(flagged)]
    G = nx.Graph(); G.add_nodes_from(flagged)
    G.add_edges_from(zip(sub.src, sub.dst))
    rings = []
    for comp in nx.connected_components(G):
        if len(comp) < min_size:
            continue
        e = sub[sub.src.isin(comp) & sub.dst.isin(comp)]
        H = nx.DiGraph(); H.add_edges_from(zip(e.src, e.dst))
        entry = [n for n in H if H.in_degree(n) == 0]
        exit_ = [n for n in H if H.out_degree(n) == 0]
        sc = None if scores is None else float(np.mean([scores[np.searchsorted(ids, m)] for m in comp]))
        rings.append({"members": sorted(int(x) for x in comp), "n_tx": int(len(e)), "volume": float(e.amount.sum()),
                      "t_start": int(e.ts.min()) if len(e) else 0, "t_end": int(e.ts.max()) if len(e) else 0,
                      "shape": shape_guess(H), "entry": sorted(int(x) for x in entry)[:6],
                      "exit": sorted(int(x) for x in exit_)[:6], "risk": sc})
    rings.sort(key=lambda r: (-(r["risk"] or 0), -len(r["members"])))
    for i, r in enumerate(rings):
        r["case_id"] = f"LL-{i + 1:03d}"
    return rings


def ring_truth_overlap(ring, sim_accounts):
    acc = sim_accounts.set_index("account_id")
    m = acc.loc[ring["members"]]
    return float(m.is_fraud.mean()), (m[m.is_fraud].ring_id.mode().iloc[0] if m.is_fraud.any() else -1)


def case_narrative(ring, reasons, truth=None):
    lines = [f"**Case {ring['case_id']}** - {len(ring['members'])} linked accounts moved "
             f"**{ring['volume']:,.0f}** across **{ring['n_tx']}** transfers between steps "
             f"{ring['t_start']} and {ring['t_end']} ({ring['t_end'] - ring['t_start'] + 1} steps).",
             f"The money-flow shape is consistent with **{ring['shape'].lower()}**."]
    if ring["entry"]:
        lines.append(f"Funds enter the network at account(s) {', '.join(map(str, ring['entry']))}.")
    if ring["exit"]:
        lines.append(f"Funds leave the network at account(s) {', '.join(map(str, ring['exit']))}.")
    if reasons:
        lines.append("Highest-risk account was flagged because it " + "; ".join(
            f"{r['text']}" for r in reasons[:3]) + ".")
    lines.append("**Suggested next steps:** (1) review KYC and account opening dates for entry and exit accounts; "
                 "(2) check counterparties outside this ring for onward movement; (3) compare timing against any "
                 "known typologies; (4) escalate for a suspicious-activity report decision.")
    if truth is not None:
        lines.append(f"_Simulator ground truth: {truth[0]:.0%} of these accounts are real mules._")
    return "\n\n".join(lines)
