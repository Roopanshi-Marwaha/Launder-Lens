"""Feature engineering shared by AMLSim, the simulator and the live replay.

Features are split into three groups so we can ablate them honestly:
  amt_*  -> anything derived from transaction amounts (the 'shortcut' group)
  str_*  -> pure graph structure (cycles, degrees, communities, ...)
  tmp_*  -> timing / activity (bursts, velocity, forwarding lag) - no amounts
"""
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp

GROUP_PREFIX = {"amount": ("amt_",), "structure": ("str_", "tmp_"), "all": ("amt_", "str_", "tmp_")}


def feature_cols(df, group):
    pre = GROUP_PREFIX[group]
    return [c for c in df.columns if c.startswith(pre)]


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def cycle_counts(A):
    """Exact numbers of directed simple 2-, 3- and 4-cycles through every node.
    A: binary sparse adjacency without self loops. Uses closed-walk algebra with the
    correction for non-simple 4-walks (verified against networkx in tests)."""
    A = sp.csr_matrix(A, dtype=np.float64)
    R = A.multiply(A.T).tocsr()
    r = np.asarray(R.sum(axis=1)).ravel()
    A2 = (A @ A).tocsr()
    c3 = np.asarray(A2.multiply(A.T).sum(axis=1)).ravel()
    closed4 = np.asarray(A2.multiply(A2.T).sum(axis=1)).ravel()
    non_simple = r ** 2 + np.asarray(R @ r).ravel() - r
    c4 = closed4 - non_simple
    return r, c3, np.maximum(c4, 0)


def compute_features(tx, accounts, init_balance=None, seed=42, return_ops=False):
    """tx: DataFrame[src,dst,amount,ts]; accounts: array-like of account ids."""
    accounts = np.asarray(accounts)
    n = len(accounts)
    idx = pd.Index(accounts)
    s = idx.get_indexer(tx["src"].values)
    d = idx.get_indexer(tx["dst"].values)
    a = tx["amount"].to_numpy(dtype=float)
    t = tx["ts"].to_numpy(dtype=float)
    F = {}

    # ---- amount group -------------------------------------------------
    out_cnt = np.bincount(s, minlength=n).astype(float)
    in_cnt = np.bincount(d, minlength=n).astype(float)
    la = np.log1p(a)

    def side(key, cnt, name):
        sm = np.bincount(key, weights=a, minlength=n)
        sq = np.bincount(key, weights=a * a, minlength=n)
        mean = _safe_div(sm, cnt)
        std = np.sqrt(np.maximum(_safe_div(sq, cnt) - mean ** 2, 0))
        lm = _safe_div(np.bincount(key, weights=la, minlength=n), cnt)
        lq = _safe_div(np.bincount(key, weights=la * la, minlength=n), cnt)
        lstd = np.sqrt(np.maximum(lq - lm ** 2, 0))
        mx = pd.Series(a).groupby(key).max().reindex(range(n)).fillna(0).to_numpy()
        F[f"amt_{name}_sum"], F[f"amt_{name}_mean"], F[f"amt_{name}_max"] = sm, mean, mx
        F[f"amt_{name}_cv"] = _safe_div(std, mean)
        F[f"amt_{name}_logstd"] = lstd
        return sm

    out_sum = side(s, out_cnt, "out")
    in_sum = side(d, in_cnt, "in")
    F["amt_net_flow"] = in_sum - out_sum
    F["amt_passthrough"] = _safe_div(np.minimum(in_sum, out_sum), np.maximum(in_sum, out_sum))
    if init_balance is not None:
        F["amt_init_balance"] = np.asarray(init_balance, dtype=float)

    # ---- graph structure ---------------------------------------------
    key = s.astype(np.int64) * n + d
    uniq, ecount = np.unique(key, return_counts=True)
    es, ed = (uniq // n).astype(int), (uniq % n).astype(int)
    keep = es != ed
    es, ed, ecount = es[keep], ed[keep], ecount[keep]
    A = sp.csr_matrix((np.ones(len(es)), (es, ed)), shape=(n, n))
    W = sp.csr_matrix((ecount.astype(float), (es, ed)), shape=(n, n))
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    in_deg = np.asarray(A.sum(axis=0)).ravel()
    F["str_out_deg"], F["str_in_deg"] = out_deg, in_deg
    F["str_deg_ratio"] = np.log((in_deg + 1) / (out_deg + 1))
    r, c3, c4 = cycle_counts(A)
    F["str_c2"], F["str_c3"], F["str_c4"] = r, c3, c4
    F["str_in_cycle"] = ((r + c3 + c4) > 0).astype(float)
    F["str_reciprocity"] = _safe_div(r, out_deg)

    Ub = ((A + A.T) > 0).astype(float).tocsr()
    Uw = (W + W.T).tocsr()
    udeg = np.asarray(Ub.sum(axis=1)).ravel()
    F["str_nbr_mean_deg"] = _safe_div(np.asarray(Ub @ udeg).ravel(), udeg)
    Ug = nx.from_scipy_sparse_array(Uw)
    Ug.remove_edges_from(nx.selfloop_edges(Ug))
    cl = nx.clustering(Ug)
    F["str_clustering"] = np.array([cl[i] for i in range(n)])
    core = nx.core_number(Ug)
    F["str_kcore"] = np.array([core[i] for i in range(n)], dtype=float)
    pr = nx.pagerank(nx.from_scipy_sparse_array(A, create_using=nx.DiGraph), alpha=0.85, max_iter=100)
    F["str_pagerank"] = np.array([pr[i] for i in range(n)]) * n
    comms = nx.community.louvain_communities(Ug, weight="weight", seed=seed)
    cid = np.zeros(n, dtype=int)
    for k, nodes in enumerate(comms):
        cid[list(nodes)] = k
    csize = np.bincount(cid, minlength=len(comms)).astype(float)
    Uc = Uw.tocoo()
    same = cid[Uc.row] == cid[Uc.col]
    inside_w = np.bincount(Uc.row[same], weights=Uc.data[same], minlength=n)
    tot_w = np.bincount(Uc.row, weights=Uc.data, minlength=n)
    c_edges = np.bincount(cid[Uc.row[same]], minlength=len(comms)) / 2.0
    c_density = _safe_div(c_edges, csize * (csize - 1) / 2.0)
    F["str_comm_size"] = csize[cid]
    F["str_comm_density"] = c_density[cid]
    F["str_comm_internal_frac"] = _safe_div(inside_w, tot_w)

    # ---- temporal / activity group -----------------------------------
    F["tmp_out_tx"], F["tmp_in_tx"] = out_cnt, in_cnt
    ev = pd.DataFrame({"acct": np.concatenate([s, d]), "t": np.concatenate([t, t])})
    g = ev.groupby(["acct", "t"]).size()
    lvl = g.groupby(level=0)
    tot = in_cnt + out_cnt
    active = lvl.size().reindex(range(n)).fillna(0).to_numpy()
    F["tmp_active_steps"] = active
    F["tmp_burstiness"] = _safe_div(lvl.max().reindex(range(n)).fillna(0).to_numpy(), tot)
    first = ev.groupby("acct").t.min().reindex(range(n)).to_numpy()
    last = ev.groupby("acct").t.max().reindex(range(n)).to_numpy()
    span = np.nan_to_num(last - first, nan=0.0) + 1
    F["tmp_span"] = span
    F["tmp_velocity"] = tot / span
    bi = pd.DataFrame({"a": d, "t": t, "o": s}).drop_duplicates().groupby(["a", "t"]).size()
    bo = pd.DataFrame({"a": s, "t": t, "o": d}).drop_duplicates().groupby(["a", "t"]).size()
    F["tmp_max_in_burst"] = bi.groupby(level=0).max().reindex(range(n)).fillna(0).to_numpy()
    F["tmp_max_out_burst"] = bo.groupby(level=0).max().reindex(range(n)).fillna(0).to_numpy()
    F["tmp_n_in_bursts"] = (bi >= 4).groupby(level=0).sum().reindex(range(n)).fillna(0).to_numpy()
    F["tmp_n_out_bursts"] = (bo >= 4).groupby(level=0).sum().reindex(range(n)).fillna(0).to_numpy()
    ins = pd.DataFrame({"acct": d, "t": t}).drop_duplicates().sort_values("t")
    outs = pd.DataFrame({"acct": s, "t_out": t}).drop_duplicates().sort_values("t_out")
    m = pd.merge_asof(ins, outs, left_on="t", right_on="t_out", by="acct", direction="forward")
    lag = (m["t_out"] - m["t"]).to_numpy()
    m["fast"] = (lag <= 2).astype(float)
    m["lag"] = np.where(np.isnan(lag), 250.0, lag)
    fm = m.groupby("acct").agg(fast=("fast", "mean"), lag=("lag", "median"))
    F["tmp_fwd_fast_frac"] = fm["fast"].reindex(range(n)).fillna(0).to_numpy()
    F["tmp_fwd_median_lag"] = fm["lag"].reindex(range(n)).fillna(250).to_numpy()

    # ---- "joined a burst" features: was *my own* transfer part of a convergence / dispersal? ----
    T1 = int(t.max()) + 2 if len(t) else 2
    ti = t.astype(np.int64)
    kin = d.astype(np.int64) * T1 + ti
    u_in = np.unique(kin * n + s)
    bk, bc = np.unique(u_in // n, return_counts=True)
    burst_in_tx = bc[np.searchsorted(bk, kin)]
    kout = s.astype(np.int64) * T1 + ti
    u_out = np.unique(kout * n + d)
    bk2, bc2 = np.unique(u_out // n, return_counts=True)
    burst_out_tx = bc2[np.searchsorted(bk2, kout)]
    F["tmp_joined_in_burst"] = pd.Series(burst_in_tx).groupby(s).max().reindex(range(n)).fillna(0).to_numpy()
    F["tmp_joined_out_burst"] = pd.Series(burst_out_tx).groupby(d).max().reindex(range(n)).fillna(0).to_numpy()

    # ---- windowed (time-respecting) short cycles: laundering loops close FAST ------------------
    tmin, tmax = int(t.min()), int(t.max())
    for W in (6, 24):
        best = np.zeros(n)
        for st in range(tmin, tmax + 1, max(1, W // 2)):
            msk = (ti >= st) & (ti < st + W) & (s != d)
            if msk.sum() < 3:
                continue
            Aw = sp.csr_matrix((np.ones(msk.sum()), (s[msk], d[msk])), shape=(n, n))
            Aw.data[:] = 1.0
            rw, c3w, c4w = cycle_counts(Aw)
            best = np.maximum(best, rw + c3w + c4w)
        F[f"str_wcycle_{W}"] = best

    out = pd.DataFrame(F, index=accounts)
    out.index.name = "account_id"
    out = out.replace([np.inf, -np.inf], 0).fillna(0)
    if not return_ops:
        return out

    def rownorm(M):
        z = np.asarray(M.sum(axis=1)).ravel()
        return sp.diags(_safe_div(np.ones(n), z)) @ M
    ops = {"out": rownorm(A).tocsr(), "in": rownorm(A.T.tocsr()).tocsr(), "und": rownorm(Ub).tocsr()}
    return out, ops
