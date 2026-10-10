import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.config import LABELS, MODELS, SEED
from src.evaluate import evaluate_probs, load_split, save_predictions

CANDIDATES = {
    "tfidf_logreg": ("TF-IDF + LogReg", [0.5, 1, 2, 4, 8],
                     lambda c: LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED)),
    "tfidf_svm": ("TF-IDF + LinearSVC", [0.05, 0.1, 0.3, 1],
                  lambda c: CalibratedClassifierCV(LinearSVC(C=c, class_weight="balanced", random_state=SEED), cv=3)),
    "tfidf_nb": ("TF-IDF + Complement NB", [0.03, 0.1, 0.3, 1],
                 lambda a: ComplementNB(alpha=a)),
}


def features():
    return FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300_000, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)),
    ])


def main():
    train, val = load_split("train"), load_split("val")
    tfidf = features()
    start = time.time()
    x_train = tfidf.fit_transform(train["text"])
    x_val = tfidf.transform(val["text"])
    print(f"признаков: {x_train.shape[1]}, векторизация {time.time() - start:.0f} с", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    for name, (title, grid, make) in CANDIDATES.items():
        search, fitted = {}, {}
        for value in grid:
            start = time.time()
            fitted[value] = make(value).fit(x_train, train["label"])
            search[value] = f1_score(val["label"], fitted[value].predict(x_val), labels=LABELS, average="macro")
            print(f"{name} {value}: val macro-F1 {search[value]:.4f} ({time.time() - start:.0f} с)", flush=True)
        best = max(search, key=search.get)
        model = Pipeline([("tfidf", tfidf), ("clf", fitted[best])])
        assert list(model.classes_) == LABELS or sorted(model.classes_) == sorted(LABELS)
        order = [list(model.classes_).index(label) for label in LABELS]
        joblib.dump(model, MODELS / f"{name}.joblib")
        probs = save_predictions(name, lambda texts: model.predict_proba(texts)[:, order])
        evaluate_probs(name, probs, title, {"param": best, "val_search": {str(k): round(v, 4) for k, v in search.items()}})


if __name__ == "__main__":
    main()
