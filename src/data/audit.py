import argparse
import csv
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.config import ANNOTATION, LABELS, PROCESSED, REPORTS, SEED

SHEET = ANNOTATION / "audit.csv"
KEY = ANNOTATION / "audit_key.csv"
DISAGREEMENTS = ANNOTATION / "audit_disagreements.csv"
SHORTCUTS = {"н": "neutral", "о": "insult", "т": "toxic", "n": "neutral", "i": "insult", "t": "toxic"}


def make(per_class):
    train = pd.read_parquet(PROCESSED / "train.parquet")
    train = train[train["source"] == "okru"]
    sample = (
        train.groupby("label")
        .sample(per_class, random_state=SEED)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    sample.insert(0, "id", range(1, len(sample) + 1))
    ANNOTATION.mkdir(parents=True, exist_ok=True)
    sample[["id", "text_raw"]].assign(manual="").to_csv(SHEET, index=False, sep=";", encoding="utf-8-sig")
    sample[["id", "label", "labels"]].to_csv(KEY, index=False, encoding="utf-8-sig")
    print(f"{len(sample)} примеров сохранено в {SHEET}")


def read_sheet():
    rows = []
    with open(SHEET, encoding="utf-8-sig", newline="") as f:
        for line in f.read().splitlines()[1:]:
            fields = next(csv.reader([line], delimiter=";"))
            if len(fields) < 3:
                fields = next(csv.reader([fields[0]]))
            rows.append({"id": int(fields[0]), "text_raw": fields[1], "manual": fields[2] if len(fields) > 2 else ""})
    return pd.DataFrame(rows)


def score():
    sheet = read_sheet().merge(pd.read_csv(KEY, encoding="utf-8-sig"), on="id")
    sheet["manual"] = sheet["manual"].astype(str).str.strip().str.lower().map(lambda v: SHORTCUTS.get(v, v))
    done = sheet[sheet["manual"].isin(LABELS)]
    agree = (done["manual"] == done["label"]).mean()
    kappa = cohen_kappa_score(done["label"], done["manual"])
    per_class = (done["manual"] == done["label"]).groupby(done["label"]).mean()
    matrix = pd.DataFrame(
        confusion_matrix(done["label"], done["manual"], labels=LABELS),
        index=[f"датасет:{c}" for c in LABELS],
        columns=[f"вручную:{c}" for c in LABELS],
    )
    result = {
        "annotated": len(done),
        "agreement": round(float(agree), 3),
        "cohen_kappa": round(float(kappa), 3),
        "agreement_by_dataset_label": {k: round(float(v), 3) for k, v in per_class.items()},
        "confusion": matrix.to_dict(),
    }
    (REPORTS / "label_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    diff = done[done["manual"] != done["label"]][["id", "text_raw", "label", "manual"]]
    verdicts = read_verdicts()
    diff = diff.assign(verdict=diff["id"].map(verdicts).fillna(""))
    diff.to_csv(DISAGREEMENTS, index=False, sep=";", encoding="utf-8-sig")
    reviewed = diff[diff["verdict"].isin(["д", "р"])]
    if len(reviewed):
        dataset_wrong = int((reviewed["verdict"] == "р").sum())
        unreviewed = len(diff) - len(reviewed)
        result["disagreements_reviewed"] = len(reviewed)
        result["dataset_label_errors"] = dataset_wrong
        result["dataset_label_accuracy"] = round(1 - (dataset_wrong + unreviewed) / len(done), 3)
    (REPORTS / "label_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "confusion"}, ensure_ascii=False, indent=2))
    print(matrix.to_string())
    print(f"{len(diff)} расхождений, проверено {len(reviewed)}, файл {DISAGREEMENTS}")


def read_verdicts():
    if not DISAGREEMENTS.exists():
        return {}
    old = pd.read_csv(DISAGREEMENTS, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    old["verdict"] = old["verdict"].str.strip().str.lower()
    return dict(zip(old["id"].astype(int), old["verdict"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["make", "score"])
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()
    make(args.per_class) if args.command == "make" else score()
