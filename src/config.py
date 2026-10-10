from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ANNOTATION = DATA / "annotation"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42

LABELS = ["neutral", "insult", "toxic"]
LABEL_NAMES_RU = {"neutral": "нейтральное", "insult": "оскорбительное", "toxic": "токсичное"}
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
MODELS = ROOT / "models"
