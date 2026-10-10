import io
import sys

from src.data.prepare import okru_label
from src.data.sources import load_2ch, load_okru, load_paradetox, load_textdetox

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def describe(name, texts, labels):
    lens, words = texts.str.len(), texts.str.split().str.len()
    print(f"\n##### {name}: {len(texts)} строк")
    print("символы: среднее %.0f, медиана %.0f, p95 %.0f, max %d" % (lens.mean(), lens.median(), lens.quantile(.95), lens.max()))
    print("слова:   среднее %.1f, медиана %.0f, p95 %.0f" % (words.mean(), words.median(), words.quantile(.95)))
    print("дубликаты текстов:", texts.duplicated().sum())
    print(labels.value_counts().to_string())


def simple_norm(texts):
    return texts.str.lower().str.split().str.join(" ")


def main():
    ok = load_okru()
    describe("ok.ru (исходные метки)", ok.text, ok.labels)
    print(ok.labels.map(okru_label).value_counts().to_string())

    ch = load_2ch()
    describe("2ch/Pikabu", ch.text, ch.toxic)

    td = load_textdetox()
    describe("TextDetox ru", td.text, td.toxic)

    pdx = load_paradetox()
    print(f"\n##### RuParaDetox: {len(pdx)} пар, уникальных токсичных {pdx.toxic_text.nunique()}")

    ok_set, ch_set = set(simple_norm(ok.text)), set(simple_norm(ch.text))
    for name, s in [("TextDetox ru", td.text), ("RuParaDetox (toxic)", pdx.toxic_text)]:
        s = simple_norm(s)
        print(f"{name}: из ok.ru {s.isin(ok_set).sum()}, из 2ch {s.isin(ch_set).sum()}, всего {len(s)}")


if __name__ == "__main__":
    main()
