"""Builds vis-network payloads (nodes/edges) for the three graph views."""
import numpy as np
import pandas as pd
from ui_kit import C

STATUS = {"tp": (C["signal"], "#ffd2c7", 15), "fn": (C["amber"], "#ffe2b8", 13), "fp": (C["violet"], "#e3d2ff", 12),
          "tn": ("#38465a", "#52627a", 6)}
STATUS_TEXT = {"tp": "caught mule", "fn": "missed mule", "fp": "false alarm", "tn": "legitimate"}


def _status(truth, flag):
    return np.where(truth & flag, "tp", np.where(truth & ~flag, "fn", np.where(~truth & flag, "fp", "tn")))


def detection_graph(tx, truth, flags, ts_max=None, neighbour_cap=110, max_interest=320, max_edges=950, seed=0):
    rng = np.random.default_rng(seed)
    t = tx if ts_max is None else tx[tx.ts <= ts_max]
    st_ = _status(truth, flags)
    active = np.zeros(len(truth), bool)
    active[np.unique(np.concatenate([t.src.values, t.dst.values]))] = True
    interest = np.flatnonzero((st_ != "tn") & active)
    pri = {"tp": 0, "fn": 1, "fp": 2}
    order = sorted(interest, key=lambda i: (pri[st_[i]], rng.random()))[:max_interest]
    keep = set(order)
    nb = t[t.src.isin(keep) | t.dst.isin(keep)]
    cand = list((set(nb.src) | set(nb.dst)) - keep)
    rng.shuffle(cand)
    keep |= set(cand[:neighbour_cap])
    e = t[t.src.isin(keep) & t.dst.isin(keep)]
    g = e.groupby(["src", "dst"]).agg(n=("amount", "size"), vol=("amount", "sum"), fr=("is_fraud", "max"), last=("ts", "max")).reset_index()
    g = g.sort_values(["fr", "vol"], ascending=False).head(max_edges)
    kept_nodes = set(order) | set(g.src) | set(g.dst)
    nodes = []
    for i in sorted(kept_nodes):
        bg, bd, sz = STATUS[st_[i]]
        nodes.append({"id": int(i), "label": str(i) if st_[i] != "tn" else "", "size": sz,
                      "color": {"background": bg, "border": bd}, "shadow": {"enabled": st_[i] != "tn", "color": bg, "size": 16, "x": 0, "y": 0},
                      "title": f"account {i}\n{STATUS_TEXT[st_[i]]}"})
    edges = [{"from": int(r.src), "to": int(r.dst), "width": 2.2 if r.fr else 0.7,
              "color": {"color": C["signal"] if r.fr else "rgba(140,155,175,.22)", "opacity": 1 if r.fr else .6},
              "title": f"{r.n} transfer(s), total {r.vol:,.0f}"} for r in g.itertuples()]
    counts = {k: int((st_[np.asarray(sorted(kept_nodes))] == k).sum()) for k in ("tp", "fn", "fp", "tn")}
    return nodes, edges, counts


def ring_graph(tx, ring):
    mem = set(ring["members"])
    e = tx[tx.src.isin(mem) & tx.dst.isin(mem)]
    g = e.groupby(["src", "dst"]).agg(n=("amount", "size"), vol=("amount", "sum"), t0=("ts", "min"), t1=("ts", "max")).reset_index()
    entry, exit_ = set(ring["entry"]), set(ring["exit"])
    nodes = []
    for m in ring["members"]:
        col = C["teal"] if m in entry else (C["amber"] if m in exit_ else C["signal"])
        role = "entry point" if m in entry else ("exit point" if m in exit_ else "intermediary")
        nodes.append({"id": int(m), "label": str(m), "size": 17, "color": {"background": col, "border": "#fff3ee"},
                      "shadow": {"enabled": True, "color": col, "size": 18, "x": 0, "y": 0}, "title": f"account {m}\n{role}"})
    mx = max(g.vol.max(), 1) if len(g) else 1
    edges = [{"from": int(r.src), "to": int(r.dst), "width": 1.2 + 4.0 * np.log1p(r.vol) / np.log1p(mx),
              "color": {"color": "rgba(255,120,90,.85)"}, "title": f"{r.n} transfer(s), {r.vol:,.0f} total\nsteps {r.t0}-{r.t1}"}
             for r in g.itertuples()]
    return nodes, edges
