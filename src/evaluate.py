import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from src.config import FIGURES, LABEL_NAMES_RU, LABELS, MODELS, PROCESSED, REPORTS, ROOT
from src.data.profanity import mask_deep
from src.data.text import normalize

METRICS = REPORTS / "metrics"
PREDICTIONS = MODELS / "predictions"
PROBES = ROOT / "resources" / "probes.csv"
EVAL_SETS = ("val", "test", "external_2ch", "probes")


def load_split(name):
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def load_probes():
    df = pd.read_csv(PROBES)
    df["text_raw"] = df["text"]
    df["text"] = df["text"].map(normalize)
    return df


def load_eval_set(name):
    return load_probes() if name == "probes" else load_split(name)


def save_predictions(name, predict_proba):
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    probs = {s: np.asarray(predict_proba(load_eval_set(s)["text"].tolist()), dtype=np.float32) for s in EVAL_SETS}
    np.savez_compressed(PREDICTIONS / f"{name}.npz", **probs)
    return probs


def load_predictions(name):
    with np.load(PREDICTIONS / f"{name}.npz") as data:
        return {s: data[s] for s in EVAL_SETS}


def to_labels(probs):
    return np.asarray(LABELS)[np.asarray(probs).argmax(1)]


def multiclass_metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro")), 4),
        "accuracy": round(float(np.mean(np.asarray(y_true) == np.asarray(y_pred))), 4),
        "per_class": {
            label: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4), "support": int(s[i])}
            for i, label in enumerate(LABELS)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def plot_confusion(matrix, title, path):
    matrix = np.asarray(matrix)
    norm = matrix / matrix.sum(axis=1, keepdims=True)
    names = [LABEL_NAMES_RU[l] for l in LABELS]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{norm[i, j]:.0%}\n{matrix[i, j]}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(LABELS)), names)
    ax.set_yticks(range(len(LABELS)), names)
    ax.set_xlabel("предсказание")
    ax.set_ylabel("истинный класс")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_labels(name, labels_by_set, title=None, extra=None):
    test = load_split("test")
    ext = load_split("external_2ch")
    probes = load_probes()
    result = {"model": name, "val_macro_f1": None, "test": multiclass_metrics(test["label"], labels_by_set["test"])}
    if "val" in labels_by_set:
        val = load_split("val")
        result["val_macro_f1"] = round(float(f1_score(val["label"], labels_by_set["val"], labels=LABELS, average="macro")), 4)

    ext_pred = np.asarray(labels_by_set["external_2ch"]) != "neutral"
    result["external_2ch_binary"] = {
        "f1_toxic": round(float(f1_score(ext["toxic"], ext_pred)), 4),
        "macro_f1": round(float(f1_score(ext["toxic"], ext_pred, average="macro")), 4),
        "size": len(ext),
    }

    probes["pred"] = labels_by_set["probes"]
    probes["correct"] = probes["pred"] == probes["label"]
    result["probes"] = {
        group: {"accuracy": round(float(g["correct"].mean()), 3), "n": len(g)}
        for group, g in probes.groupby("group", sort=False)
    }
    result["probes_errors"] = mask_deep(probes.loc[~probes["correct"], ["group", "text_raw", "label", "pred"]].to_dict("records"))
    result.update(extra or {})

    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (METRICS / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_confusion(result["test"]["confusion"], title or name, FIGURES / f"confusion_{name}.png")

    print(classification_report(test["label"], labels_by_set["test"], labels=LABELS, digits=4))
    print("val macro-F1:", result["val_macro_f1"], "| 2ch:", result["external_2ch_binary"])
    print("пробы:", {g: v["accuracy"] for g, v in result["probes"].items()})
    return result


def evaluate_probs(name, probs, title=None, extra=None):
    return evaluate_labels(name, {s: to_labels(p) for s, p in probs.items()}, title, extra)
