# Launder Lens 2.0 - red team vs blue team for graph-based money-laundering detection

**Research question:** *Can fraud detectors recognise laundering they have never seen, and does graph structure help once the amount "shortcut" is removed?*

Launder Lens benchmarks detectors on the IBM **AMLSim** data (Kaggle), then builds its own **simulation laboratory** (5 laundering typologies, legitimate decoys, two attacker knobs) to attack and harden them, and wraps everything in an investigation app.

## Run it
```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app/app.py                  # results + models are already included
```
Re-create every number from scratch (about 5-10 minutes on one CPU):
```bash
python -m src.run_pipeline            # add --quick for a 2-3 minute version
python -m pytest -q                   # 7 tests, incl. cycle counts verified against networkx
```
AMLSim CSVs live in `data/raw/` (`transactions.csv`, `accounts.csv`, `alerts.csv`).

## What is in the box
| Path | Purpose |
|---|---|
| `src/features.py` | 3 feature groups: amount `amt_*`, structure `str_*`, timing `tmp_*`. Exact 2/3/4-cycle counts via sparse algebra |
| `src/simulator.py` | Financial Crime Simulation Lab: background bank, decoys (merchants, payroll, friend circles), 5 typologies, `camouflage` and `patience` attack knobs |
| `src/models.py` | Rules, Random Forest, XGBoost, SIGN-style graph neural network, TreeSHAP |
| `src/experiments.py` | The four experiments below |
| `src/evidence.py` | Ring extraction, shape classification, template-based case narrative (no LLM, cannot hallucinate) |
| `app/` | Streamlit platform: Mission Control, Simulation Lab, Time Machine, Case Desk, Evidence Lab |

## Experiments and headline results
1. **AMLSim benchmark** - ring-grouped 5-fold CV. Hybrid XGBoost F1 **0.90** (PR-AUC 0.95); structure-first F1 0.72; amount-only 0.84; rules 0.33. SHAP weight: amounts 51%, structure 36%, timing 13% - a warning that part of the score is a simulator shortcut.
2. **Red vs blue** - disguise amounts and amount-only PR-AUC drops **0.88 → 0.19**, the hybrid **0.96 → 0.38**. Structure-first stays at ~0.75; adversarially hardened hybrid recovers to **0.74**.
3. **Unseen typologies** (leave-one-typology-out) - structural signals transfer unevenly (fan-out/smurfing well, unseen circular patterns poorly). Reported honestly.
4. **Detection latency** - snapshot replay over 200 steps; structure-first detects 85% of rings (median ~22 steps) where the amount-reliant hybrid detects 12% once amounts are camouflaged.

## Honest limitations
Synthetic data only; simulator assumptions are mine (it is used for robustness tests, AMLSim is the external benchmark); AMLSim graph features are transductive; the GNN is SIGN-style (precomputed message passing + MLP), not end-to-end GraphSAGE; only two attack knobs.

## 6-minute presentation flow
1. Hook: "My model scored F1 0.90. Then I proved it was partly cheating." 2. SHAP shows the amount shortcut. 3. Live: Simulation Lab - flip Naive → Camouflaged and watch the hybrid fall. 4. Evidence Lab heatmap: hardening vs patience. 5. Time Machine replay. 6. Case Desk: open a ring, show reasons + money trace. 7. Limits slide.
