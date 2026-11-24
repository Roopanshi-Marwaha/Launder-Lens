"""Central paths and constants for Launder Lens."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
for _d in (PROCESSED_DIR, REPORTS_DIR, MODELS_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
TYPOLOGIES = ["cycle", "fan_in", "fan_out", "smurfing", "layering"]
TYPOLOGY_LABELS = {
    "cycle": "Circular transfers", "fan_in": "Fan-in (collector)",
    "fan_out": "Fan-out (dispersal)", "smurfing": "Smurfing (split & merge)",
    "layering": "Layering (mule chain)",
}
