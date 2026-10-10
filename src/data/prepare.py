import json
import zlib
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABEL2ID, LABELS, PROCESSED, REPORTS, SEED
from src.data.sources import load_2ch, load_okru, load_paradetox, load_textdetox
from src.data.text import has_contacts, has_cyrillic, normalize

SPLITS = ("train", "val", "test")
MIN_CHARS = 2
NEAR_DUP_WORDS = 8
NEAR_DUP_MIN_WORDS = 6


def okru_label(labels):
    tags = set(labels.split(","))
    if tags & {"THREAT", "OBSCENITY"}:
        return "toxic"
    if "INSULT" in tags:
        return "insult"
    return "neutral"


def hash_split(key):
    bucket = zlib.crc32(key.encode("utf-8")) % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def near_duplicate_key(norm):
    words = norm.split()
    return " ".join(words[:NEAR_DUP_WORDS]) if len(words) >= NEAR_DUP_MIN_WORDS else norm


def drop_conflicts_and_duplicates(df, key):
    conflicts = df.groupby(key)["label"].nunique()
    conflicting = conflicts[conflicts > 1].index
    before = len(df)
    df = df[~df[key].isin(conflicting)].drop_duplicates(key)
    return df, int(len(conflicting)), before - len(df)


def prepare_okru(stats):
    df = load_okru()
    df["label"] = df["labels"].map(okru_label)
    df["norm"] = df["text"].map(normalize)
    stats["okru_raw"] = len(df)
    df = df[df["norm"].str.len() >= MIN_CHARS]
    stats["okru_after_empty"] = len(df)
    df = df[df["norm"].map(has_cyrillic)]
    stats["okru_after_no_cyrillic"] = len(df)
    contacts = df["text"].map(has_contacts)
    stats["okru_contacts_removed"] = int(contacts.sum())
    df = df[~contacts]
    df, stats["okru_conflicting_texts"], stats["okru_exact_duplicates_removed"] = drop_conflicts_and_duplicates(df, "norm")
    df["near_key"] = df["norm"].map(near_duplicate_key)
    df, stats["okru_conflicting_near_groups"], stats["okru_near_duplicates_removed"] = drop_conflicts_and_duplicates(df, "near_key")
    df = df.drop(columns="near_key")
    stats["okru_clean"] = len(df)

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    parts = [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    return pd.concat(parts).assign(source="okru", group=lambda d: d["norm"])


def prepare_paradetox(okru, stats):
    df = load_paradetox()
    df["group"] = df["toxic_text"].map(normalize)
    df["norm"] = df["neutral_text"].map(normalize)
    df = df[df["norm"].str.len() >= MIN_CHARS]
    split_of = dict(zip(okru["norm"].map(near_duplicate_key), okru["split"]))
    df["split"] = df["group"].map(lambda g: split_of.get(near_duplicate_key(g)) or hash_split(g))
    stats["paradetox_pairs"] = len(df)
    return pd.DataFrame({
        "text": df["neutral_text"],
        "norm": df["norm"],
        "label": "neutral",
        "labels": "",
        "split": df["split"],
        "source": "paradetox",
        "group": df["group"],
    })


class FragmentIndex:
    def __init__(self, texts):
        self.texts = list(texts)
        self.index = defaultdict(list)
        for i, text in enumerate(self.texts):
            words = text.split()
            for j in range(len(words) - 2):
                self.index[" ".join(words[j:j + 3])].append(i)

    def contains(self, text):
        words = text.split()
        if len(words) < 3:
            return False
        return any(text in self.texts[i] for i in self.index.get(" ".join(words[:3]), []))


def prepare_external(name, df, seen, fragments, stats):
    df = df.assign(norm=df["text"].map(normalize))
    df = df[(df["norm"].str.len() >= MIN_CHARS) & df["norm"].map(has_cyrillic)].drop_duplicates("norm")
    stats[f"{name}_clean"] = len(df)
    overlap = df["norm"].map(near_duplicate_key).isin(seen) | df["norm"].map(fragments.contains)
    stats[f"{name}_overlap_with_main"] = int(overlap.sum())
    df = df[~overlap]
    stats[f"{name}_without_overlap"] = len(df)
    return df.rename(columns={"text": "text_raw", "norm": "text"})[["text", "text_raw", "toxic"]]


def main():
    stats = {}
    okru = prepare_okru(stats)
    para = prepare_paradetox(okru, stats)

    main_df = pd.concat([okru, para], ignore_index=True)
    before = len(main_df)
    main_df = main_df.drop_duplicates("norm")
    stats["cross_source_duplicates_removed"] = before - len(main_df)

    leaks = main_df.groupby("group")["split"].nunique()
    assert (leaks <= 1).all()

    main_df["label_id"] = main_df["label"].map(LABEL2ID)
    main_df = main_df.rename(columns={"text": "text_raw", "norm": "text"})
    columns = ["text", "text_raw", "label", "label_id", "labels", "source", "group"]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        part = main_df[main_df["split"] == split][columns].sample(frac=1, random_state=SEED)
        part.to_parquet(PROCESSED / f"{split}.parquet", index=False)

    seen = set(main_df["text"].map(near_duplicate_key))
    fragments = FragmentIndex(main_df["text"])
    prepare_external("textdetox", load_textdetox(), seen, fragments, stats)
    external = prepare_external("2ch", load_2ch(), seen, fragments, stats)
    external.to_parquet(PROCESSED / "external_2ch.parquet", index=False)
    (PROCESSED / "external_textdetox.parquet").unlink(missing_ok=True)

    table = main_df.pivot_table(index="split", columns="label", values="text", aggfunc="count", fill_value=0)
    table = table.reindex(index=list(SPLITS), columns=LABELS)
    by_source = main_df.pivot_table(index="split", columns="source", values="text", aggfunc="count", fill_value=0)
    stats["splits"] = {s: {k: int(v) for k, v in table.loc[s].items()} for s in SPLITS}
    stats["splits_by_source"] = {s: {k: int(v) for k, v in by_source.loc[s].items()} for s in SPLITS}

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(table.assign(total=table.sum(axis=1)).to_string())


if __name__ == "__main__":
    main()
