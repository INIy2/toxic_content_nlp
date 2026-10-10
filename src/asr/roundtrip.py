import argparse
import asyncio
import io
import json
import re
import sys

import edge_tts
import jiwer
import pandas as pd

from src.asr.golos_wer import digits_to_words
from src.asr.whisper import Transcriber, load_audio
from src.config import RAW, REPORTS, SEED
from src.data.profanity import mask_deep
from src.data.text import normalize
from src.error_analysis import MAT
from src.evaluate import load_split, multiclass_metrics
from src.predict import ToxicityClassifier

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
VOICES = ["ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"]
AUDIO_DIR = RAW / "roundtrip"


async def synthesize(items, concurrency=8):
    semaphore = asyncio.Semaphore(concurrency)

    async def one(i, text, voice):
        path = AUDIO_DIR / f"{i:04d}.mp3"
        if path.exists() and path.stat().st_size > 0:
            return
        async with semaphore:
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(text, voice).save(str(path))
                    return
                except Exception:
                    await asyncio.sleep(2 * (attempt + 1))

    await asyncio.gather(*(one(i, t, v) for i, t, v in items))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    test = load_split("test")
    test = test[(test["source"] == "okru") & test["text"].str.split().str.len().between(3, 40)]
    sample = test.groupby("label").sample(args.per_class, random_state=SEED).reset_index(drop=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(synthesize([(i, t, VOICES[i % 2]) for i, t in enumerate(sample["text"])]))
    paths = [AUDIO_DIR / f"{i:04d}.mp3" for i in range(len(sample))]
    ok = [p.exists() and p.stat().st_size > 0 for p in paths]
    sample = sample[ok].reset_index(drop=True)
    paths = [p for p, k in zip(paths, ok) if k]

    audios = [load_audio(p) for p in paths]
    sample["seconds"] = [len(a) / 16000 for a in audios]
    sample["asr_raw"] = Transcriber()(audios)
    sample["asr"] = sample["asr_raw"].map(lambda t: normalize(digits_to_words(t)))

    mat_rx = re.compile(MAT)
    mat_ref = sample["text"].astype(object).map(lambda t: bool(mat_rx.search(t)))
    mat_kept = sample.loc[mat_ref, "asr"].astype(object).map(lambda t: bool(mat_rx.search(t)))
    censored = sample["asr_raw"].str.contains(r"\*")

    classifier = ToxicityClassifier()

    def predict(texts):
        return [r["label"] for r in classifier.predict(texts)]

    clean = multiclass_metrics(sample["label"], predict(sample["text"].tolist()))
    asr = multiclass_metrics(sample["label"], predict(sample["asr"].tolist()))
    result = {
        "synthesized": len(sample),
        "audio_minutes": round(float(sample["seconds"].sum()) / 60, 1),
        "voices": VOICES,
        "wer": round(jiwer.wer(sample["text"].tolist(), sample["asr"].tolist()), 4),
        "cer": round(jiwer.cer(sample["text"].tolist(), sample["asr"].tolist()), 4),
        "wer_by_class": {l: round(jiwer.wer(g["text"].tolist(), g["asr"].tolist()), 4) for l, g in sample.groupby("label")},
        "mat_texts": int(mat_ref.sum()),
        "mat_preserved_share": round(float(mat_kept.mean()), 4),
        "censored_with_asterisks": int(censored.sum()),
        "ensemble_on_clean_text": {"macro_f1": clean["macro_f1"], "accuracy": clean["accuracy"]},
        "ensemble_on_asr_text": {"macro_f1": asr["macro_f1"], "accuracy": asr["accuracy"]},
        "examples": mask_deep(sample.loc[sample["text"] != sample["asr"], ["label", "text", "asr"]].head(20).to_dict("records")),
    }
    (REPORTS / "metrics" / "asr_roundtrip.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    sample.drop(columns=["group"]).to_csv(AUDIO_DIR / "roundtrip.csv", index=False, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
