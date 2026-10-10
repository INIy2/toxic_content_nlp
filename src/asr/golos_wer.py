import argparse
import io
import json
import re
import sys
import time

import jiwer
import pandas as pd
import pyarrow.parquet as pq
from num2words import num2words

from src.asr.whisper import Transcriber, load_audio
from src.config import RAW, REPORTS, SEED
from src.data.text import normalize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
SUBSETS = {"crowd": RAW / "golos" / "crowd_test_1.parquet", "farfield": RAW / "golos" / "farfield_test.parquet"}


def sample(path, n):
    df = pq.read_table(path).to_pandas()
    df = df[df["transcription"].str.split().str.len() >= 2]
    return df.sample(n, random_state=SEED).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["openai/whisper-large-v3-turbo", "openai/whisper-small"])
    parser.add_argument("--per-subset", type=int, default=300)
    args = parser.parse_args()

    data = {name: sample(path, args.per_subset) for name, path in SUBSETS.items()}
    for df in data.values():
        df["audio_array"] = df["audio"].map(lambda a: load_audio(a["bytes"]))
        df["seconds"] = df["audio_array"].map(len) / 16000

    result = {"per_subset": args.per_subset, "models": {}}
    transcripts = []
    for model in args.models:
        transcribe = Transcriber(model)
        transcribe([data["crowd"]["audio_array"][0]])
        stats, examples = {}, []
        for name, df in data.items():
            start = time.time()
            hyp = transcribe(df["audio_array"].tolist())
            elapsed = time.time() - start
            refs = df["transcription"].map(normalize).tolist()
            hyps = [normalize(h) for h in hyp]
            hyps_words = [normalize(digits_to_words(h)) for h in hyp]
            stats[name] = {
                "wer": round(jiwer.wer(refs, hyps), 4),
                "wer_digits_as_words": round(jiwer.wer(refs, hyps_words), 4),
                "cer_digits_as_words": round(jiwer.cer(refs, hyps_words), 4),
                "exact_match_digits_as_words": round(sum(r == h for r, h in zip(refs, hyps_words)) / len(refs), 4),
                "audio_minutes": round(float(df["seconds"].sum()) / 60, 1),
                "real_time_factor": round(elapsed / float(df["seconds"].sum()), 4),
            }
            examples += [{"subset": name, "ref": r, "hyp": h} for r, h in zip(refs, hyps_words) if r != h][:8]
            transcripts += [{"model": model, "subset": name, "ref": r, "hyp": h} for r, h in zip(refs, hyp)]
            print(model, name, stats[name], flush=True)
        result["models"][model] = {"subsets": stats, "error_examples": examples}
        del transcribe

    (REPORTS / "metrics").mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics" / "asr_golos.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(transcripts).to_csv(RAW / "golos" / "transcripts.csv", index=False, encoding="utf-8")


def digits_to_words(text):
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="ru"), text)


if __name__ == "__main__":
    main()
