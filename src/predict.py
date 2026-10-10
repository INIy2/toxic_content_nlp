import argparse
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

from src.config import LABEL_NAMES_RU, LABELS, MODELS
from src.data.text import normalize
from src.ensemble import apply_offsets
from src.evaluate import METRICS, load_split
from src.models.neural import ARCHITECTURES, Vocab, predict_proba

AUDIO_SUFFIXES = {".ogg", ".oga", ".opus", ".mp3", ".wav", ".m4a", ".flac"}


class ToxicityClassifier:
    def __init__(self, ensemble="ensemble"):
        stages = json.loads((METRICS / f"{ensemble}.json").read_text(encoding="utf-8"))["stages"]
        self.weights = {name: w for name, w in stages["weights"].items() if w > 0}
        self.offsets = np.array([0.0, stages["offsets"]["insult"], stages["offsets"]["toxic"]])
        self.threshold = stages["doubtful"]["threshold"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vocab = None
        self.parts = {}
        for name in self.weights:
            if name.startswith("tfidf"):
                model = joblib.load(MODELS / f"{name}.joblib")
                order = [list(model.classes_).index(label) for label in LABELS]
                self.parts[name] = lambda texts, m=model, o=order: m.predict_proba(texts)[:, o]
            else:
                vocab = vocab or Vocab(load_split("train")["text"])
                net = ARCHITECTURES[name.split("_")[0]](torch.zeros(len(vocab.words), 300)).to(device)
                net.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location=device))
                self.parts[name] = lambda texts, n=net, v=vocab: predict_proba(n, v, texts, 100)

    def predict_proba(self, texts):
        clean = [normalize(t) for t in texts]
        probs = sum(w * self.parts[name](clean) for name, w in self.weights.items())
        return apply_offsets(probs, self.offsets)

    def predict(self, texts):
        results = []
        for probs in self.predict_proba(texts):
            label = LABELS[int(probs.argmax())]
            results.append({
                "label": label,
                "confidence": float(probs.max()),
                "doubtful": bool(probs.max() < self.threshold),
                "probs": {l: float(p) for l, p in zip(LABELS, probs)},
            })
        return results


def describe(title, text, result):
    verdict = "сомнительно" if result["doubtful"] else LABEL_NAMES_RU[result["label"]]
    probs = ", ".join(f"{LABEL_NAMES_RU[l]} {p:.0%}" for l, p in result["probs"].items())
    lines = [f"  {title}: {text}"]
    lines.append(f"  итог: {verdict}")
    if result["doubtful"]:
        lines.append(f"  скорее всего: {LABEL_NAMES_RU[result['label']]}")
    lines.append(f"  вероятности: {probs}")
    return "\n".join(lines)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", help="фразы или пути к аудиофайлам")
    args = parser.parse_args()

    print("загрузка моделей...")
    classifier = ToxicityClassifier()
    transcriber = None

    def handle(item):
        nonlocal transcriber
        path = Path(item)
        if path.suffix.lower() in AUDIO_SUFFIXES and path.exists():
            from src.asr.whisper import Transcriber, load_audio
            transcriber = transcriber or Transcriber()
            text = transcriber([load_audio(path)])[0]
            print(f"\n{path.name}")
            print(describe("распознано", text, classifier.predict([text])[0]))
        else:
            print()
            print(describe("фраза", item, classifier.predict([item])[0]))

    if args.inputs:
        for item in args.inputs:
            handle(item)
        return

    print("пишите фразу или путь к голосовому (.ogg, .mp3, .wav), пустая строка для выхода")
    while True:
        try:
            item = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not item:
            break
        handle(item.strip('"'))


if __name__ == "__main__":
    main()
