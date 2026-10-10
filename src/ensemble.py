import argparse
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from src.config import FIGURES, LABELS, SEED
from src.evaluate import METRICS, evaluate_labels, load_predictions, load_split

DEFAULT_MODELS = ["tfidf_logreg", "tfidf_svm", "tfidf_nb", "textcnn_navec", "bigru_navec"]
DOUBTFUL = "doubtful"
LABEL_IDS = {label: i for i, label in enumerate(LABELS)}


def macro_f1(y, probs):
    return f1_score(y, probs.argmax(1), labels=range(len(LABELS)), average="macro")


def blend(preds, weights, split):
    return sum(w * preds[m][split] for m, w in weights.items())


def search_weights(preds, y_val, samples):
    names = list(preds)
    rng = np.random.default_rng(SEED)
    candidates = list(np.eye(len(names))) + [np.full(len(names), 1 / len(names))] + list(rng.dirichlet(np.ones(len(names)), samples))
    best_w, best_f1 = None, -1.0
    for w in candidates:
        f1 = macro_f1(y_val, sum(wi * preds[n]["val"] for wi, n in zip(w, names)))
        if f1 > best_f1:
            best_w, best_f1 = w, f1
    for _ in range(3):
        for i in range(len(names)):
            for delta in np.linspace(-0.1, 0.1, 9):
                w = best_w.copy()
                w[i] = max(0.0, w[i] + delta)
                if w.sum() == 0:
                    continue
                w = w / w.sum()
                f1 = macro_f1(y_val, sum(wi * preds[n]["val"] for wi, n in zip(w, names)))
                if f1 > best_f1:
                    best_w, best_f1 = w, f1
    return {n: round(float(w), 4) for n, w in zip(names, best_w)}, best_f1


def search_offsets(probs, y_val):
    logp = np.log(probs + 1e-9)
    grid = np.round(np.arange(-1.5, 1.51, 0.1), 2)
    best, best_f1 = (0.0, 0.0), macro_f1(y_val, logp)
    for a, b in itertools.product(grid, grid):
        f1 = macro_f1(y_val, logp + np.array([0.0, a, b]))
        if f1 > best_f1:
            best, best_f1 = (float(a), float(b)), f1
    return np.array([0.0, *best]), best_f1


def apply_offsets(probs, offsets):
    logits = np.log(probs + 1e-9) + offsets
    exp = np.exp(logits - logits.max(1, keepdims=True))
    return exp / exp.sum(1, keepdims=True)


def doubtful_curve(probs, y, thresholds):
    conf = probs.max(1)
    pred = probs.argmax(1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append({
            "threshold": round(float(t), 2),
            "doubtful_share": round(float(1 - keep.mean()), 4),
            "macro_f1_confident": round(float(f1_score(y[keep], pred[keep], labels=range(len(LABELS)), average="macro")), 4),
            "errors_caught": round(float(((pred != y) & ~keep).sum() / max((pred != y).sum(), 1)), 4),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--name", default="ensemble")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--doubtful-share", type=float, default=0.05)
    args = parser.parse_args()

    preds = {m: load_predictions(m) for m in args.models}
    y_val = load_split("val")["label_id"].values
    y_test = load_split("test")["label_id"].values

    single = {m: {"val": round(macro_f1(y_val, p["val"]), 4), "test": round(macro_f1(y_test, p["test"]), 4)} for m, p in preds.items()}
    weights, val_blend = search_weights(preds, y_val, args.samples)
    blended = {s: blend(preds, weights, s) for s in ("val", "test", "external_2ch", "probes")}
    offsets, val_offsets = search_offsets(blended["val"], y_val)
    adjusted = {s: apply_offsets(p, offsets) for s, p in blended.items()}

    thresholds = np.round(np.arange(0.4, 0.96, 0.05), 2)
    conf_val = adjusted["val"].max(1)
    tau = float(np.quantile(conf_val, args.doubtful_share))
    curve_test = doubtful_curve(adjusted["test"], y_test, thresholds)
    at_tau = doubtful_curve(adjusted["test"], y_test, [tau])[0]

    stages = {
        "single_models": single,
        "weights": weights,
        "blend": {"val": round(val_blend, 4), "test": round(macro_f1(y_test, blended["test"]), 4)},
        "offsets": {"insult": round(offsets[1], 2), "toxic": round(offsets[2], 2)},
        "blend_offsets": {"val": round(val_offsets, 4), "test": round(macro_f1(y_test, adjusted["test"]), 4)},
        "doubtful": {"target_share_val": args.doubtful_share, "threshold": round(tau, 4), "test": at_tau, "curve_test": curve_test},
    }
    print(json.dumps({k: v for k, v in stages.items() if k != "doubtful"}, ensure_ascii=False, indent=2))
    print("сомнительно:", at_tau)

    labels = {s: np.asarray(LABELS)[p.argmax(1)] for s, p in adjusted.items()}
    evaluate_labels(args.name, labels, "Ансамбль + постобработка", {"stages": stages})

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    shares = [r["doubtful_share"] * 100 for r in curve_test]
    ax1.plot(shares, [r["macro_f1_confident"] for r in curve_test], "o-", color="#4C72B0", label="macro-F1 на уверенных")
    ax1.set_xlabel("доля сомнительных сообщений, %")
    ax1.set_ylabel("macro-F1", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(shares, [r["errors_caught"] * 100 for r in curve_test], "s--", color="#C8453B", label="перехвачено ошибок")
    ax2.set_ylabel("доля пойманных ошибок, %", color="#C8453B")
    ax1.axvline(at_tau["doubtful_share"] * 100, color="gray", linestyle=":")
    ax1.set_title('Метка "сомнительно" на test')
    fig.tight_layout()
    fig.savefig(FIGURES / f"{args.name}_doubtful.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
