<div align="center">

# 🔍 Launder Lens 2.0

### Red team vs blue team for graph-based money-laundering detection

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-189AB4)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NetworkX](https://img.shields.io/badge/Graphs-NetworkX-2C7BB6)
![Tests](https://img.shields.io/badge/tests-7%20passed-brightgreen)

*Can fraud detectors recognise laundering they have never seen, and does graph structure help once the amount "shortcut" is removed?*

</div>

---

## Overview

Most fraud models look great on a benchmark and then quietly fall apart when the attacker changes behaviour. **Launder Lens** puts that to the test.

It benchmarks detectors on the IBM **AMLSim** data (Kaggle), then builds its own **Financial Crime Simulation Lab** with 5 laundering typologies, legitimate decoys and two attacker knobs, so we can attack and harden the detectors like a real red team / blue team exercise. Everything is wrapped in an interactive **investigation app** for exploring results, replaying detection over time and opening individual cases.

## Key findings

| # | Finding | Result |
|---|---|---|
| 1 | A hybrid XGBoost model scores very high on AMLSim | **F1 0.90**, PR-AUC 0.95 |
| 2 | But SHAP shows it leans on amounts | Amounts **51%**, structure 36%, timing 13% |
| 3 | Disguise the amounts (red team) and amount-based models collapse | Hybrid PR-AUC **0.96 → 0.38**, amount-only **0.88 → 0.19** |
| 4 | Structure-first detection holds up | PR-AUC stays at **~0.75** |
| 5 | Adversarial hardening recovers the hybrid | Back up to **0.74** |
| 6 | Structure-first catches rings far earlier under camouflage | **85%** of rings detected (median ~22 steps) vs **12%** for the amount-reliant hybrid |

> **Takeaway:** a model that scored F1 0.90 was partly cheating through an amount shortcut. Graph structure is what survives when attackers adapt.

## How it works

```
AMLSim data ──────────┐
                      ├──> Feature groups (amount / structure / timing)
Simulation Lab ───────┘            │
(5 typologies, decoys,             ▼
 camouflage + patience)    Detectors: Rules, RF, XGBoost, SIGN GNN
                                   │
                                   ▼
                    Experiments: benchmark, red vs blue,
                    unseen typologies, detection latency
                                   │
                                   ▼
                    Evidence: ring extraction + case narrative
                                   │
                                   ▼
                       Streamlit investigation app
```

## The investigation app

| Page | What you can do |
|---|---|
| **Mission Control** | See the headline results at a glance |
| **Simulation Lab** | Flip the attacker from Naive to Camouflaged and watch the detectors react live |
| **Time Machine** | Replay detection step by step over a simulated timeline |
| **Case Desk** | Open a suspicious ring, read the reasons and trace the money |
| **Evidence Lab** | Explore hardening vs patience across the full robustness grid |

## Quick start

```bash
git clone https://github.com/Roopanshi-Marwaha/Launder-Lens.git
cd Launder-Lens
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app/app.py                  # results + models are already included
```

The app opens at `http://localhost:8501`. On Windows you can also run `.\run_app.bat`, and on Linux/Mac `./run_app.sh`.

## Experiments

1. **AMLSim benchmark** - ring-grouped 5-fold cross-validation. Hybrid XGBoost F1 **0.90** (PR-AUC 0.95), structure-first F1 0.72, amount-only 0.84, rules 0.33.
2. **Red vs blue** - the attacker disguises amounts; detectors are retrained with adversarial examples to recover.
3. **Unseen typologies** (leave-one-typology-out) - structural signals transfer well to fan-out/smurfing patterns and less well to unseen circular patterns.
4. **Detection latency** - snapshot replay over 200 steps, measuring how quickly each detector finds a ring.

## Re-create every number from scratch

About 5-10 minutes on one CPU:

```bash
python -m src.run_pipeline            # add --quick for a 2-3 minute version
python -m pytest -q                   # 7 tests, incl. cycle counts verified against networkx
```

## Data

The raw AMLSim CSVs are **not** stored in this repository because of their size. To re-run the full pipeline:

1. Download the IBM AMLSim dataset from Kaggle.
2. Place `transactions.csv`, `accounts.csv` and `alerts.csv` in `data/raw/`.

The app itself does not need the raw data: trained models, saved scenarios and results are already included.

## Project structure

```
Launder-Lens/
├── app/                  Streamlit investigation platform
├── src/
│   ├── features.py       amount / structure / timing features, exact 2-3-4 cycle counts
│   ├── simulator.py      Financial Crime Simulation Lab
│   ├── models.py         Rules, Random Forest, XGBoost, SIGN-style GNN, TreeSHAP
│   ├── experiments.py    The four experiments
│   ├── evidence.py       Ring extraction, shape classification, case narrative
│   └── run_pipeline.py   End-to-end runner
├── models/               Trained lab models
├── artifacts/            Saved simulation scenarios (naive, patient, camouflaged)
├── reports/              Experiment outputs (CSV / JSON) used by the app
├── tests/                Test suite
└── data/raw/             AMLSim CSVs go here (not tracked)
```

## Highlights

- **Three feature groups** (`amt_*`, `str_*`, `tmp_*`) so every result can be traced to what the model actually relied on.
- **Exact 2/3/4-cycle counts** via sparse algebra, verified against NetworkX in the test suite.
- **Two attacker knobs**, `camouflage` (hide amounts) and `patience` (slow the laundering down), to stress-test detectors.
- **Explainable cases**: ring extraction and shape classification feed a template-based narrative, with no LLM involved, so it cannot hallucinate.
- **Ring-grouped cross-validation** so rings never leak between train and test folds.

## Contributors

| Name | GitHub |
|---|---|
| Roopanshi Marwaha | [@Roopanshi-Marwaha](https://github.com/Roopanshi-Marwaha) |
| Kashish Singla | [@SKashhh](https://github.com/SKashhh) |

## Data source

IBM AMLSim synthetic transaction data, as distributed on Kaggle. All credit for the dataset goes to its original authors.
