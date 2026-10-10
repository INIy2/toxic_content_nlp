import io
import re
import zipfile

import pandas as pd
import requests

from src.config import RAW

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/"
HF_PARQUET = "https://huggingface.co/api/datasets/{}/parquet/default/{}/0.parquet"


def _kaggle(ref, member):
    path = RAW / member
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        resp = requests.get(KAGGLE + ref, timeout=600)
        resp.raise_for_status()
        zipfile.ZipFile(io.BytesIO(resp.content)).extract(member, RAW)
    return path


def _hf(dataset, split, name):
    path = RAW / f"{name}_{split}.parquet"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(HF_PARQUET.format(dataset, split)).to_parquet(path, index=False)
    return pd.read_parquet(path)


def load_okru():
    rows = []
    with open(_kaggle("alexandersemiletov/toxic-russian-comments", "dataset.txt"), encoding="utf-8") as f:
        for line in f:
            labels = re.findall(r"__label__(\w+)", line)
            text = re.sub(r"(__label__\w+,?)+\s*", "", line, count=1).strip()
            rows.append((text, ",".join(labels)))
    return pd.DataFrame(rows, columns=["text", "labels"])


def load_2ch():
    df = pd.read_csv(_kaggle("blackmoon/russian-language-toxic-comments", "labeled.csv"))
    return pd.DataFrame({"text": df.comment.str.strip(), "toxic": df.toxic.astype(int)})


def load_textdetox():
    df = _hf("textdetox/multilingual_toxicity_dataset", "ru", "textdetox")
    return df[["text", "toxic"]]


def load_paradetox():
    parts = [_hf("s-nlp/ru_paradetox", s, "paradetox").assign(split=s) for s in ("train", "validation")]
    return pd.concat(parts, ignore_index=True).rename(
        columns={"ru_toxic_comment": "toxic_text", "ru_neutral_comment": "neutral_text"}
    )
