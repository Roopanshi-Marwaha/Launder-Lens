"""Financial Crime Simulation Laboratory.

Generates a configurable synthetic bank network with known ground truth:
  * a legitimate background (communities, popular merchants, heavy-tailed amounts)
  * legitimate DECOYS that look suspicious (merchant sale-day bursts, payroll fan-out,
    friend groups that send money in circles)
  * five laundering typologies injected with ring ids and timing
Two red-team knobs let an attacker evade the detectors:
  camouflage in [0,1] - disguise amounts as normal traffic (breaks amount shortcuts)
  patience   in [0,1] - slow the laundering down (breaks timing signals)
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .config import TYPOLOGIES


@dataclass
class SimConfig:
    n_accounts: int = 3000
    n_steps: int = 200
    avg_tx: float = 14.0
    rings_per_type: int = 8
    typologies: tuple = tuple(TYPOLOGIES)
    camouflage: float = 0.0
    patience: float = 0.0
    decoys: bool = True
    mule_dormancy: float = 0.5   # fraction of a mule's ordinary background traffic that never happens
    seed: int = 0


@dataclass
class Sim:
    cfg: SimConfig
    tx: pd.DataFrame
    accounts: pd.DataFrame
    rings: pd.DataFrame


def _background(cfg, rng):
    n, T = cfg.n_accounts, cfg.n_steps
    comm = rng.integers(0, max(2, n // 40), n)
    order = np.argsort(comm, kind="stable")
    ncomm = comm.max() + 1
    csize = np.bincount(comm, minlength=ncomm)
    cstart = np.concatenate([[0], np.cumsum(csize)[:-1]])
    pop = rng.pareto(1.7, n) + 1
    pop /= pop.sum()
    rate = rng.lognormal(0, 0.6, n)
    n_a = rng.poisson(rate / rate.mean() * cfg.avg_tx)
    src = np.repeat(np.arange(n), n_a)
    m = len(src)
    local = rng.random(m) < 0.55
    c = comm[src]
    dst_local = order[cstart[c] + (rng.random(m) * csize[c]).astype(int)]
    dst_glob = rng.choice(n, size=m, p=pop)
    dst = np.where(local, dst_local, dst_glob)
    dst = np.where(dst == src, (dst + 1) % n, dst)
    w_len = rng.integers(60, T + 1, n)
    w_start = (rng.random(n) * (T - w_len + 1)).astype(int)
    ts = w_start[src] + (rng.random(m) * w_len[src]).astype(int)
    scale = rng.lognormal(np.log(150), 0.9, n)
    amount = scale[src] * rng.lognormal(0, 0.8, m)
    df = pd.DataFrame({"src": src, "dst": dst, "amount": np.round(amount, 2), "ts": ts})
    bal = np.round(rng.lognormal(np.log(5000), 1.0, n), 2)
    return df, bal, pop, scale


def _decoys(cfg, rng, pop, kind, taken):
    n, T = cfg.n_accounts, cfg.n_steps
    rows = []

    def add(s, d, a, t):
        rows.append(pd.DataFrame({"src": s, "dst": d, "amount": np.round(a, 2), "ts": t}))
    merchants = np.argsort(-pop)[: max(3, int(0.015 * n))]
    kind[merchants] = "merchant"
    for mch in merchants:                                   # sale-day bursts (legit fan-in)
        for _ in range(3):
            k = int(rng.integers(6, 19))
            cust = rng.choice(n, k, replace=False)
            cust = cust[cust != mch]
            add(cust, np.full(len(cust), mch), 60 * rng.lognormal(0, 0.7, len(cust)),
                np.full(len(cust), int(rng.integers(0, T))))
    free = np.flatnonzero(kind == "none")
    for emp in rng.choice(free, max(3, int(0.015 * n)), replace=False):   # payroll (legit fan-out)
        kind[emp] = "payroll"
        staff = rng.choice(n, int(rng.integers(6, 17)), replace=False)
        staff = staff[staff != emp]
        base = rng.lognormal(np.log(2500), 0.3)
        period = int(rng.integers(20, 41))
        for t0 in range(int(rng.integers(0, period)), T, period):
            add(np.full(len(staff), emp), staff, base * (1 + rng.normal(0, 0.02, len(staff))),
                np.full(len(staff), t0))
    for _ in range(max(3, n // 40)):                        # friend groups (legit circular flows)
        k = int(rng.integers(2, 5))
        grp = rng.choice(n, k, replace=False)
        m = int(rng.integers(6, 15))
        a = rng.integers(0, k, m)
        b = (a + rng.integers(1, k, m)) % k
        add(grp[a], grp[b], 80 * rng.lognormal(0, 0.7, m), rng.integers(0, T, m))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["src", "dst", "amount", "ts"])


def _ring_amounts(cfg, rng):
    """Returns f(hop, base_total_split) giving amounts under the camouflage setting."""
    c = cfg.camouflage
    naive = rng.lognormal(np.log(4000), 0.4)
    legit = rng.lognormal(np.log(150), 0.9) * rng.lognormal(0, 0.5)
    log_base = (1 - c) * np.log(naive) + c * np.log(legit)
    sigma = (1 - c) * 0.02 + c * 0.7
    skim = (1 - c) * 0.03

    def amt(hop=0, scale=1.0, size=None):
        return np.exp(log_base + rng.normal(0, sigma, size)) * ((1 - skim) ** hop) * scale
    return amt


def _inject(cfg, rng, pool_iter):
    T = cfg.n_steps
    stretch = 1 + cfg.patience * 6
    rows, rings, rid = [], [], 0
    for typ in cfg.typologies:
        for _ in range(cfg.rings_per_type):
            amt = _ring_amounts(cfg, rng)
            lag = lambda lo, hi: max(0, int(round(rng.integers(lo, hi + 1) * stretch)))
            tx = []   # (src, dst, amount, ts)
            if typ == "cycle":
                k = int(rng.integers(3, 7)); mem = pool_iter(k); laps = int(rng.integers(1, 3))
                gap = lag(0, 2); t0 = int(rng.integers(0, max(1, T - laps * k * (gap + 1) - 2)))
                for lp in range(laps):
                    for i in range(k):
                        tx.append((mem[i], mem[(i + 1) % k], float(amt(lp * k + i)), t0 + (lp * k + i) * max(gap, 1 if stretch > 1 else gap)))
            elif typ == "fan_in":
                m = int(rng.integers(6, 15)); mem = pool_iter(m + 2); w = lag(0, 2); lg = lag(1, 3)
                t0 = int(rng.integers(0, max(1, T - w - lg - 2)))
                for i in range(m):
                    tx.append((mem[i], mem[m], float(amt(0)), t0 + int(rng.integers(0, w + 1))))
                tx.append((mem[m], mem[m + 1], float(amt(1, scale=m)), t0 + w + lg))
            elif typ == "fan_out":
                m = int(rng.integers(6, 15)); mem = pool_iter(m + 1); w = lag(0, 2)
                t0 = int(rng.integers(0, max(1, T - w - 2)))
                for i in range(m):
                    tx.append((mem[0], mem[1 + i], float(amt(0)), t0 + int(rng.integers(0, w + 1))))
            elif typ == "smurfing":
                m = int(rng.integers(4, 9)); mem = pool_iter(m + 2); w = lag(0, 1); lg = lag(1, 3)
                t0 = int(rng.integers(0, max(1, T - w - lg - 2)))
                for i in range(m):
                    t1 = t0 + int(rng.integers(0, w + 1))
                    tx.append((mem[0], mem[1 + i], float(amt(0)), t1))
                    tx.append((mem[1 + i], mem[m + 1], float(amt(1)), t1 + lg + int(rng.integers(0, 2))))
            else:  # layering
                L = int(rng.integers(4, 9)); mem = pool_iter(L); t = int(rng.integers(0, max(1, T - L * 3 - 2)))
                for i in range(L - 1):
                    tx.append((mem[i], mem[i + 1], float(amt(i)), t)); t += lag(1, 3)
            tx = [(s, d, a, min(int(t), T - 1)) for s, d, a, t in tx]
            members = sorted({x for s, d, _, _ in tx for x in (s, d)})
            tss = [t for *_, t in tx]
            for s, d, a, t in tx:
                rows.append((s, d, round(a, 2), t, True, rid, typ))
            rings.append({"ring_id": rid, "typology": typ, "members": members,
                          "start_ts": min(tss), "end_ts": max(tss), "n_tx": len(tx)})
            rid += 1
    return rows, rings


def simulate(cfg: SimConfig) -> Sim:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_accounts
    bg, bal, pop, _ = _background(cfg, rng)
    kind = np.array(["none"] * n, dtype=object)
    parts = [bg]
    if cfg.decoys:
        parts.append(_decoys(cfg, rng, pop, kind, None))
    legit = pd.concat(parts, ignore_index=True)
    legit["is_fraud"], legit["ring_id"], legit["typology"] = False, -1, "none"
    pool = list(rng.permutation(np.flatnonzero(kind == "none")))
    taken = []

    def pool_iter(k):
        got = [pool.pop() for _ in range(k)]
        taken.extend(got)
        return [int(x) for x in got]
    rows, rings = _inject(cfg, rng, pool_iter)
    if cfg.mule_dormancy > 0 and taken:
        mule = np.zeros(n, dtype=bool); mule[taken] = True
        touch = mule[legit.src.values] | mule[legit.dst.values]
        drop = touch & (rng.random(len(legit)) < cfg.mule_dormancy)
        legit = legit[~drop]
    fr = pd.DataFrame(rows, columns=["src", "dst", "amount", "ts", "is_fraud", "ring_id", "typology"])
    tx = pd.concat([legit, fr], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    tx["tx_id"] = np.arange(len(tx))
    tx[["src", "dst", "ts", "ring_id"]] = tx[["src", "dst", "ts", "ring_id"]].astype(int)
    acc = pd.DataFrame({"account_id": np.arange(n), "init_balance": bal, "is_fraud": False,
                        "ring_id": -1, "typology": "none", "role": kind})
    rings_df = pd.DataFrame(rings)
    for r in rings:
        acc.loc[r["members"], ["is_fraud", "ring_id", "typology"]] = [True, r["ring_id"], r["typology"]]
    return Sim(cfg, tx, acc, rings_df)
