import numpy as np, networkx as nx, scipy.sparse as sp, pytest
from src.features import cycle_counts, compute_features, feature_cols
from src.simulator import simulate, SimConfig
from src.models import best_threshold, metrics, TabModel
from src.evidence import find_rings


def test_cycle_counts_match_networkx():
    rng = np.random.default_rng(3); n = 35
    M = (rng.random((n, n)) < 0.13).astype(float); np.fill_diagonal(M, 0)
    r, c3, c4 = cycle_counts(sp.csr_matrix(M))
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    ref = {2: np.zeros(n), 3: np.zeros(n), 4: np.zeros(n)}
    for c in nx.simple_cycles(G, length_bound=4):
        for v in c:
            ref[len(c)][v] += 1
    assert np.allclose(r, ref[2]) and np.allclose(c3, ref[3]) and np.allclose(c4, ref[4])


def test_simulator_ground_truth_consistent():
    sim = simulate(SimConfig(n_accounts=800, rings_per_type=3, seed=1))
    members = {m for ms in sim.rings.members for m in ms}
    assert set(sim.accounts[sim.accounts.is_fraud].account_id) == members
    assert sim.tx.ts.min() >= 0 and sim.tx.ts.max() < 200
    assert (sim.tx.src != sim.tx.dst).all()
    assert sim.tx[sim.tx.is_fraud].ring_id.nunique() == len(sim.rings)


def test_camouflage_removes_amount_signal():
    def gap(c):
        s = simulate(SimConfig(n_accounts=1500, camouflage=c, seed=2))
        f = np.log1p(s.tx.groupby("src").amount.max().reindex(s.accounts.account_id).fillna(0))
        y = s.accounts.is_fraud.values
        return f[y].mean() - f[~y].mean()
    assert gap(0.0) > 2 * abs(gap(1.0))


def test_features_have_three_groups_and_no_labels():
    sim = simulate(SimConfig(n_accounts=600, rings_per_type=2, seed=3))
    f = compute_features(sim.tx, sim.accounts.account_id.values, sim.accounts.init_balance.values)
    assert feature_cols(f, "amount") and feature_cols(f, "structure")
    assert not [c for c in f.columns if c.split("_")[-1] in ("fraud", "label") or "alert" in c or "ring_id" in c or "is_fraud" in c]
    assert np.isfinite(f.values).all()
    assert not set(feature_cols(f, "amount")) & set(feature_cols(f, "structure"))


def test_structure_model_ignores_amounts():
    sim = simulate(SimConfig(n_accounts=900, rings_per_type=4, seed=4))
    f, ops = compute_features(sim.tx, sim.accounts.account_id.values, return_ops=True)
    y = sim.accounts.is_fraud.values.astype(int)
    m = TabModel("structure").fit([dict(df=f, ops=ops, y=y)])
    assert not any(c.startswith("amt_") for c in m.cols)
    assert 0 <= m.predict(f).min() and m.predict(f).max() <= 1


def test_metrics_and_threshold():
    y = np.array([0, 0, 0, 1, 1]); p = np.array([.1, .2, .3, .8, .9])
    t = best_threshold(y, p); r = metrics(y, p, t)
    assert r["f1"] == 1.0 and r["tp"] == 2


def test_find_rings_groups_connected_flagged_accounts():
    sim = simulate(SimConfig(n_accounts=700, rings_per_type=2, seed=5))
    flags = sim.accounts.is_fraud.values
    rings = find_rings(sim.tx, flags, sim.accounts.account_id.values)
    assert rings and all(len(r["members"]) >= 3 for r in rings)
