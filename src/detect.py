"""Glue used by the app: run the trained lab models on any simulated network."""
import numpy as np
import pandas as pd
from .features import compute_features
from .models import metrics
from .experiments import ring_recall


def analyze(sim, bundle):
    feats, ops = compute_features(sim.tx, sim.accounts.account_id.values, sim.accounts.init_balance.values, return_ops=True)
    scores = {k: m.predict(feats, ops) for k, m in bundle["models"].items()}
    flags = {k: scores[k] >= bundle["thr"][k] for k in scores}
    return dict(feats=feats, ops=ops, scores=scores, flags=flags)


def compare(sim, an):
    y = sim.accounts.is_fraud.values.astype(int)
    rows = []
    for k in an["scores"]:
        r = metrics(y, an["scores"][k], _thr(an, k))
        r.update(model=k, ring_recall=ring_recall(sim, an["flags"][k]), flagged=int(an["flags"][k].sum()))
        rows.append(r)
    return pd.DataFrame(rows)


def _thr(an, k):
    s = an["scores"][k]; f = an["flags"][k]
    return float(s[f].min()) if f.any() else float(s.max() + 1)


def typology_recovery(sim, flags):
    rows = []
    for t, g in sim.rings.groupby("typology"):
        rec = [flags[m].mean() >= 0.5 for m in g.members]
        cov = [flags[m].mean() for m in g.members]
        rows.append(dict(typology=t, rings=len(g), rings_found=int(sum(rec)), ring_recall=float(np.mean(rec)),
                         member_coverage=float(np.mean(cov))))
    return pd.DataFrame(rows)
