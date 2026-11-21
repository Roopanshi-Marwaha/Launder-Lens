# Launder Lens 2.0

**Red team vs blue team for graph-based money-laundering detection**

> **Research question:** *Can fraud detectors recognise laundering they have never seen, and does graph structure help once the amount "shortcut" is removed?*

Launder Lens benchmarks detectors on the IBM **AMLSim** data (Kaggle), then builds its own **simulation laboratory** (5 laundering typologies, legitimate decoys, two attacker knobs) to attack and harden them, and wraps everything in an interactive investigation app.

## Contributors

| Name | GitHub |
|---|---|
| Roopanshi Marwaha | [@Roopanshi-Marwaha](https://github.com/Roopanshi-Marwaha) |
| Kashish Singla | [@SKashhh](https://github.com/SKashhh) |

## Quick start

```bash
git clone https://github.com/Roopanshi-Marwaha/Launder-Lens.git
cd Launder-Lens
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app/app.py                  # results + models are already included
```

The app opens at `http://localhost:8501`. On Windows you can also run `.\run_app.bat`, and on Linux/Mac `./run_app.sh`.

Tested on Python 3.14.

## Data

The raw AMLSim CSVs are **not** stored in this repository because of their size. To re-run the full pipeline:

1. Download the IBM AMLSim dataset from Kaggle.
2. Place `transactions.csv`, `accounts.csv` and `alerts.csv` in `data/raw/`.

The app itself does not need the raw data: trained models, saved scenarios and results are already included.

## Re-create every number from scratch

About 5-10 minutes on one CPU:

```bash
python -m src.run_pipeline            # add --quick for a 2-3 minute version
python -m pytest -q                   # 7 tests, incl. cycle counts verified against networkx
```

## What is in the box

| Path | Purpose |
|---|---|
| `src/features.py` | 3 feature groups: amount `amt_*`, structure `str_*`, timing `tmp_*`. Exact 2/3/4-cycle counts via sparse algebra |
| `src/simulator.py` | Financial Crime Simulation Lab: background bank, decoys (merchants, payroll, friend circles), 5 typologies, `camouflage` and `patience` attack knobs |
| `src/models.py` | Rules, Random Forest, XGBoost, SIGN-style graph neural network, TreeSHAP |
| `src/experiments.py` | The four experiments below |
| `src/evidence.py` | Ring extraction, shape classification, template-based case narrative (no LLM, cannot hallucinate) |
| `src/run_pipeline.py` | Runs everything end to end |
| `app/` | Streamlit platform: Mission Control, Simulation Lab, Time Machine, Case Desk, Evidence Lab |
| `models/` | Trained lab models |
| `artifacts/` | Saved simulation scenarios (naive, patient, camouflaged) |
| `reports/` | Experiment outputs (CSV/JSON) used by the app |
| `tests/` | Test suite |

## Experiments and headline results

1. **AMLSim benchmark** - ring-grouped 5-fold CV. Hybrid XGBoost F1 **0.90** (PR-AUC 0.95); structure-first F1 0.72; amount-only 0.84; rules 0.33. SHAP weight: amounts 51%, structure 36%, timing 13%. This is a warning that part of the score is a simulator shortcut.
2. **Red vs blue** - disguise amounts and amount-only PR-AUC drops **0.88 → 0.19**, the hybrid **0.96 → 0.38**. Structure-first stays at ~0.75; the adversarially hardened hybrid recovers to **0.74**.
3. **Unseen typologies** (leave-one-typology-out) - structural signals transfer unevenly (fan-out/smurfing well, unseen circular patterns poorly). Reported honestly.
4. **Detection latency** - snapshot replay over 200 steps; structure-first detects 85% of rings (median ~22 steps) where the amount-reliant hybrid detects 12% once amounts are camouflaged.

## Honest limitations

- Synthetic data only.
- Simulator assumptions are ours. It is used for robustness tests; AMLSim is the external benchmark.
- AMLSim graph features are transductive.
- The GNN is SIGN-style (precomputed message passing + MLP), not end-to-end GraphSAGE.
- Only two attack knobs.

## 6-minute presentation flow

1. Hook: "Our model scored F1 0.90. Then we proved it was partly cheating."
2. SHAP shows the amount shortcut.
3. Live: Simulation Lab - flip Naive → Camouflaged and watch the hybrid fall.
4. Evidence Lab heatmap: hardening vs patience.
5. Time Machine replay.
6. Case Desk: open a ring, show reasons + money trace.
7. Limits slide.

## Data source

IBM AMLSim synthetic transaction data, as distributed on Kaggle. All credit for the dataset goes to its original authors.