"""All experiments of the project. Every number shown in the report / app is produced here."""
import json, pickle, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .config import TYPOLOGIES, REPORTS_DIR, MODELS_DIR, ARTIFACTS_DIR, PROCESSED_DIR
from .data_loader import load_amlsim, amlsim_groups_and_typology
from .features import compute_features
from .models import TabModel, RuleModel, SignGNN, best_threshold, metrics
from .simulator import simulate, SimConfig


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ helpers
def make_graph(c, p, seed, typs=TYPOLOGIES, rings=8, n=3000):
    sim = simulate(SimConfig(n_accounts=n, camouflage=c, patience=p, seed=seed,
                             typologies=tuple(typs), rings_per_type=rings))
    df, ops = compute_features(sim.tx, sim.accounts.account_id.values, sim.accounts.init_balance.values, return_ops=True)
    return dict(df=df, ops=ops, y=sim.accounts.is_fraud.values.astype(int), sim=sim)


def ents(gs):
    return [dict(df=g["df"], ops=g["ops"], y=g["y"]) for g in gs]


def base_suite(seed=0):
    return {"rules": RuleModel(), "xgb_amount": TabModel("amount", seed=seed), "xgb_structure": TabModel("structure", seed=seed),
            "xgb_all": TabModel("all", seed=seed), "gnn_structure": SignGNN("structure", seed=seed)}


def thresholds(models, val):
    out = {}
    for k, m in models.items():
        p = np.concatenate([m.predict(g["df"], g["ops"]) for g in val])
        out[k] = best_threshold(np.concatenate([g["y"] for g in val]), p)
    return out


def ring_recall(sim, flags):
    if len(sim.rings) == 0:
        return float("nan")
    return float(np.mean([flags[m].mean() >= 0.5 for m in sim.rings.members]))


def evaluate(models, thr, g, extra=None):
    rows = []
    for k, m in models.items():
        p = m.predict(g["df"], g["ops"])
        r = metrics(g["y"], p, thr[k])
        r.update(model=k, ring_recall=ring_recall(g["sim"], p >= thr[k]), **(extra or {}))
        rows.append(r)
    return rows


# ------------------------------------------------------------------ A. AMLSim benchmark
def exp_amlsim(n_splits=5, seed=42):
    cache = PROCESSED_DIR / "amlsim_features.pkl"
    tx, acc, alerts = load_amlsim()
    if cache.exists():
        feats, ops = pickle.load(open(cache, "rb"))
    else:
        log("AMLSim: building features ...")
        feats, ops = compute_features(tx, acc.account_id.values, acc.init_balance.values, return_ops=True)
        pickle.dump((feats, ops), open(cache, "wb"))
    y = acc.is_fraud.astype(int).values
    groups, typ = amlsim_groups_and_typology(tx, acc, alerts)
    models_f = lambda: {"rules": RuleModel(), "rf_all": TabModel("all", "rf", seed), "xgb_amount": TabModel("amount", seed=seed),
                        "xgb_structure": TabModel("structure", seed=seed), "xgb_all": TabModel("all", seed=seed),
                        "gnn_structure": SignGNN("structure", seed), "gnn_all": SignGNN("all", seed)}
    rows, typ_rows, shap_rows = [], [], []
    skf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(feats, y, groups)):
        itr, iva = next(StratifiedGroupKFold(5, shuffle=True, random_state=fold).split(tr, y[tr], groups[tr]))
        mtr = np.zeros(len(y), bool); mtr[tr[itr]] = True
        mva = np.zeros(len(y), bool); mva[tr[iva]] = True
        models = models_f()
        for name, m in models.items():
            m.fit([dict(df=feats, ops=ops, y=y, mask=mtr)])
            p = m.predict(feats, ops)
            thr = best_threshold(y[mva], p[mva])
            r = metrics(y[te], p[te], thr); r.update(model=name, fold=fold)
            rows.append(r)
            pred = p[te] >= thr
            for tname in ("cycle", "fan_in"):
                sel = typ[te] == tname
                typ_rows.append(dict(model=name, fold=fold, typology=tname, n=int(sel.sum()),
                                     hit=int((pred & sel).sum())))
            if name == "xgb_all":
                c = m.contribs(feats.iloc[te]).drop(columns="bias").abs().mean()
                shap_rows.append(c)
        log(f"AMLSim fold {fold + 1}/{n_splits} done")
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / "amlsim_cv_folds.csv", index=False)
    summ = res.groupby("model")[["ap", "precision", "recall", "f1"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(REPORTS_DIR / "amlsim_cv_summary.csv", index=False)
    t = pd.DataFrame(typ_rows).groupby(["model", "typology"])[["n", "hit"]].sum().reset_index()
    t["recall"] = t.hit / t.n
    t.to_csv(REPORTS_DIR / "amlsim_per_typology.csv", index=False)
    sh = pd.concat(shap_rows, axis=1).mean(axis=1).sort_values(ascending=False).rename("mean_abs_shap").reset_index()
    sh.columns = ["feature", "mean_abs_shap"]
    sh["group"] = sh.feature.str.split("_").str[0].map({"amt": "amount", "str": "structure", "tmp": "timing"})
    sh.to_csv(REPORTS_DIR / "amlsim_shap.csv", index=False)
    return summ


# ------------------------------------------------------------------ B. Red team vs blue team
def exp_robustness(seeds=(100, 200), c_grid=(0, .25, .5, .75, 1), p_grid=(0, .5, 1)):
    log("Robustness: building training graphs ...")
    naive_tr, naive_va = [make_graph(0, 0, 1)], [make_graph(0, 0, 2)]
    hard_specs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    hard_tr = [make_graph(c, p, 10 + i) for i, (c, p) in enumerate(hard_specs)]
    hard_va = [make_graph(c, p, 20 + i) for i, (c, p) in enumerate(hard_specs)]
    models = base_suite()
    for m in models.values():
        m.fit(ents(naive_tr))
    thr = thresholds(models, naive_va)
    hard = {"xgb_all_hardened": TabModel("all"), "xgb_structure_hardened": TabModel("structure"),
            "gnn_structure_hardened": SignGNN("structure")}
    for m in hard.values():
        m.fit(ents(hard_tr))
    thr.update(thresholds(hard, hard_va))
    models.update(hard)
    rows = []
    for sd in seeds:
        for c in c_grid:
            for p in p_grid:
                g = make_graph(c, p, sd + int(c * 4) * 7 + int(p * 2))
                rows += evaluate(models, thr, g, dict(camouflage=c, patience=p, seed=sd))
        log(f"Robustness seed {sd} done")
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / "robustness_grid.csv", index=False)
    return res


# ------------------------------------------------------------------ C. Unseen typologies
def exp_loto(seeds=(300, 400), camouflage=(0.0, 1.0)):
    rows = []
    for c in camouflage:
        for held in TYPOLOGIES:
            seen = [t for t in TYPOLOGIES if t != held]
            tr = [make_graph(c, 0, 50, seen)]; va = [make_graph(c, 0, 51, seen)]
            models = base_suite()
            for m in models.values():
                m.fit(ents(tr))
            thr = thresholds(models, va)
            for sd in seeds:
                te = make_graph(c, 0, sd + TYPOLOGIES.index(held), [held], rings=14)
                rows += evaluate(models, thr, te, dict(camouflage=c, held_out=held, seed=sd))
            log(f"LOTO c={c} held={held} done")
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / "loto_results.csv", index=False)
    return res


# ------------------------------------------------------------------ D. Temporal replay / detection latency
def snapshot(sim, t):
    tx = sim.tx[sim.tx.ts <= t]
    df, ops = compute_features(tx, sim.accounts.account_id.values, sim.accounts.init_balance.values, return_ops=True)
    fr = sim.tx[sim.tx.is_fraud & (sim.tx.ts <= t)]
    y = np.zeros(len(sim.accounts), int)
    y[np.unique(np.concatenate([fr.src.values, fr.dst.values]))] = 1
    return dict(df=df, ops=ops, y=y)


def snap_entries(specs, ts):
    out = []
    for c, p, sd in specs:
        g = make_graph(c, p, sd)
        out += [snapshot(g["sim"], t) for t in ts]
    return out


def ring_latency(sim, snap_ts, flags):
    rows = []
    for _, r in sim.rings.iterrows():
        det = np.nan
        for i, t in enumerate(snap_ts):
            if t >= r.start_ts and flags[i][r.members].mean() >= 0.5:
                det = t; break
        rows.append(dict(ring_id=r.ring_id, typology=r.typology, start_ts=r.start_ts, end_ts=r.end_ts,
                         detect_ts=det, latency=det - r.start_ts if not np.isnan(det) else np.nan))
    return pd.DataFrame(rows)


def exp_temporal(test_snaps=None):
    test_snaps = test_snaps or list(range(9, 200, 10))
    train_ts = [49, 99, 149, 199]
    log("Temporal: training on snapshots ...")
    struct = {"struct": TabModel("structure"), "gnn": SignGNN("structure")}
    tr_s = snap_entries([(0, 0, 31), (1, 0, 32)], train_ts); va_s = snap_entries([(1, 0, 42)], train_ts)
    for m in struct.values():
        m.fit(tr_s)
    allm = {"all": TabModel("all")}
    tr_a = snap_entries([(0, 0, 31)], train_ts); va_a = snap_entries([(0, 0, 41)], train_ts)
    allm["all"].fit(tr_a)
    thr = thresholds(struct, va_s); thr.update(thresholds(allm, va_a))
    models = {**struct, **allm}
    lat, curves = [], []
    for name, (c, p) in {"naive": (0, 0), "camouflaged": (1, 0), "patient": (1, 0.5)}.items():
        g = make_graph(c, p, 500)
        sim = g["sim"]
        scores = {k: [] for k in models}
        y_t = []
        for t in test_snaps:
            s = snapshot(sim, t); y_t.append(s["y"])
            for k, m in models.items():
                scores[k].append(m.predict(s["df"], s["ops"]).astype(np.float32))
        scores = {k: np.array(v) for k, v in scores.items()}
        for k in models:
            flags = scores[k] >= thr[k]
            l = ring_latency(sim, test_snaps, flags); l["model"], l["scenario"] = k, name
            lat.append(l)
            for i, t in enumerate(test_snaps):
                yt = y_t[i]
                curves.append(dict(scenario=name, model=k, ts=t, flagged=int(flags[i].sum()),
                                   true_flagged=int((flags[i] & (yt == 1)).sum()), false_flagged=int((flags[i] & (yt == 0)).sum()),
                                   fraud_so_far=int(yt.sum())))
        pickle.dump(dict(sim=sim, snap_ts=test_snaps, scores=scores, thr=thr, y_t=np.array(y_t), c=c, p=p),
                    open(ARTIFACTS_DIR / f"scenario_{name}.pkl", "wb"))
        log(f"Temporal scenario {name} done")
    pd.concat(lat).to_csv(REPORTS_DIR / "ring_latency.csv", index=False)
    pd.DataFrame(curves).to_csv(REPORTS_DIR / "replay_curves.csv", index=False)


# ------------------------------------------------------------------ E. Models used by the live lab
def build_lab_bundle():
    import joblib
    log("Lab models: training ...")
    specs = [(0, 0), (1, 0), (0, 1), (1, 1), (.5, .5)]
    tr = [make_graph(c, p, 700 + i) for i, (c, p) in enumerate(specs)]
    va = [make_graph(c, p, 800 + i) for i, (c, p) in enumerate(specs)]
    naive_tr = [tr[0]]; naive_va = [va[0]]
    models = {"hybrid_naive": TabModel("all"), "structure_hardened": TabModel("structure"),
              "hybrid_hardened": TabModel("all")}
    models["hybrid_naive"].fit(ents(naive_tr))
    models["structure_hardened"].fit(ents(tr)); models["hybrid_hardened"].fit(ents(tr))
    thr = {"hybrid_naive": thresholds({"m": models["hybrid_naive"]}, naive_va)["m"],
           "structure_hardened": thresholds({"m": models["structure_hardened"]}, va)["m"],
           "hybrid_hardened": thresholds({"m": models["hybrid_hardened"]}, va)["m"]}
    joblib.dump({"models": models, "thr": thr}, MODELS_DIR / "lab_models.joblib")
    return thr


def write_headline():
    h = {}
    try:
        a = pd.read_csv(REPORTS_DIR / "amlsim_cv_summary.csv").set_index("model")
        h["amlsim"] = a[["f1_mean", "ap_mean", "recall_mean"]].round(3).to_dict("index")
        sh = pd.read_csv(REPORTS_DIR / "amlsim_shap.csv")
        h["amlsim_shap_share"] = (sh.groupby("group").mean_abs_shap.sum() / sh.mean_abs_shap.sum()).round(3).to_dict()
        r = pd.read_csv(REPORTS_DIR / "robustness_grid.csv")
        g = r.groupby(["model", "camouflage", "patience"])[["ap", "f1", "recall", "ring_recall"]].mean().reset_index()
        h["robustness"] = {m: {"clean_ap": float(g[(g.model == m) & (g.camouflage == 0) & (g.patience == 0)].ap.iloc[0]),
                               "camo_ap": float(g[(g.model == m) & (g.camouflage == 1) & (g.patience == 0)].ap.iloc[0]),
                               "worst_ap": float(g[g.model == m].ap.min())} for m in g.model.unique()}
    except Exception as e:  # pragma: no cover
        h["error"] = str(e)
    (REPORTS_DIR / "headline.json").write_text(json.dumps(h, indent=2))
    return h
