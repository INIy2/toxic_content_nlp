import json
import re

import numpy as np
import pandas as pd

from src.config import LABELS
from src.ensemble import apply_offsets, blend
from src.evaluate import METRICS, load_predictions, load_split

MAT = r"\b(?:бля\w*|хуй\w*|хуе\w*|хуя\w*|пизд\w*|еба\w*|ебу\w*|ебл\w*|ебн\w*|сука|суки|сучк\w*|нахуй|похуй)\b"
VIOLENCE = r"\b(?:уби\w*|убь\w*|расстрел\w*|растрел\w*|стрел\w*|повес\w*|вешать|сжечь|сжига\w*|кастрир\w*|казни\w*|четверт\w*|кол\b|колу\b|на кол|отруб\w*|утопи\w*|задуши\w*|мочить|замоч\w*|уничтож\w*|истреб\w*|резать|зареж\w*|сдохн\w*|сдох\w*|подох\w*|придуш\w*|разорв\w*|закопа\w*|прибь\w*|прибить|бить|избить|набить|сломаю|сломать)"
SEXUAL = r"\b(?:сос\w*|отсос\w*|трах\w*|выеб\w*|ебать|выебать|отъеб\w*|член\w*|в жопу|в рот|очко|дырк\w*|изнасил\w*|насилов\w*|залуп\w*)"
ADDRESS = r"\b(?:ты|тебя|тебе|тобой|твой|твоя|твое|твои|вы|вас|вам|вами)\b"
POLITICS = r"\b(?:путин\w*|депутат\w*|власт\w*|правительств\w*|госдум\w*|едрос\w*|единорос\w*|чиновник\w*|навальн\w*|украин\w*|хохл\w*|укроп\w*|пенси\w*|президент\w*|либерал\w*|фашист\w*|нацист\w*)\b"


def has(texts, pattern):
    rx = re.compile(pattern)
    return texts.map(lambda t: bool(rx.search(t)))


def ensemble_frame(split="test", name="ensemble"):
    stages = json.loads((METRICS / f"{name}.json").read_text(encoding="utf-8"))["stages"]
    preds = {m: load_predictions(m) for m in stages["weights"]}
    offsets = np.array([0.0, stages["offsets"]["insult"], stages["offsets"]["toxic"]])
    probs = apply_offsets(blend(preds, stages["weights"], split), offsets)
    df = load_split(split).copy()
    df["pred"] = np.asarray(LABELS)[probs.argmax(1)]
    df["confidence"] = probs.max(1)
    df["p_true"] = probs[np.arange(len(df)), df["label_id"].values]
    df["doubtful"] = df["confidence"] < stages["doubtful"]["threshold"]
    df["words"] = df["text"].str.split().str.len()
    text = df["text"].astype(object)
    for col, pattern in (("mat", MAT), ("violence", VIOLENCE), ("sexual", SEXUAL), ("address", ADDRESS), ("politics", POLITICS)):
        df[col] = has(text, pattern)
    return df


def categorize(row):
    pair = (row.label, row.pred)
    if row.words <= 2:
        return "очень короткий текст (1-2 слова)"
    if pair == ("neutral", "insult") and row.mat:
        return "мат без оскорбления, принят за оскорбление"
    if pair in (("toxic", "insult"), ("toxic", "neutral")) and row.sexual and not row.violence:
        return "сексуальное унижение, принято за оскорбление"
    if pair in (("toxic", "insult"), ("toxic", "neutral")) and not row.violence:
        return "угроза без явных слов про насилие"
    if pair == ("insult", "toxic") and (row.violence or row.sexual):
        return "оскорбление со словами про насилие, принято за угрозу"
    if pair == ("toxic", "insult") and row.violence:
        return "угроза вместе с оскорблением, принята за оскорбление"
    if pair == ("insult", "toxic"):
        return "пожелание наказания (посадить, засунуть)"
    if pair == ("insult", "neutral"):
        return "оскорбление через ярлыки или искаженный мат"
    if pair == ("neutral", "insult") and row.politics:
        return "резкое политическое высказывание без оскорбления"
    if row.label == "neutral":
        return "слова с двойным смыслом или спорная метка"
    return "прочее"


def errors_table(df):
    errors = df[df["label"] != df["pred"]].copy()
    errors["category"] = errors.apply(categorize, axis=1)
    return errors
