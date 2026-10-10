import html
import re

_URL = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:ru|com|net|org|cc|me)/\S*")
_PHONE = re.compile(r"(?:\+7|\b8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b")
_VK_MENTION = re.compile(r"\[(?:id|club)\d+\|([^\]]*)\]")
_AT_MENTION = re.compile(r"@\w+")
_TAG = re.compile(r"<[^>]+>")
_STAR_IN_WORD = re.compile(r"(?<=[а-я])\*+(?=[а-я])")
_WORD = re.compile(r"\w+")
_CYRILLIC = re.compile(r"[а-я]")
_HOMOGLYPHS = str.maketrans("aceopxykmthb", "асеорхукмтнв")
_STRETCH = re.compile(r"([а-яa-z])\1{3,}")
_NON_WORD = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")


def _fix_homoglyphs(match):
    word = match.group()
    return word.translate(_HOMOGLYPHS) if _CYRILLIC.search(word) else word


def has_contacts(text):
    return bool(_URL.search(text) or _PHONE.search(text))


def has_cyrillic(text):
    return bool(_CYRILLIC.search(text))


def normalize(text):
    text = html.unescape(str(text))
    text = _TAG.sub(" ", text)
    text = _VK_MENTION.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _AT_MENTION.sub(" ", text)
    text = text.lower().replace("ё", "е")
    text = _STAR_IN_WORD.sub("", text)
    text = _WORD.sub(_fix_homoglyphs, text)
    text = _STRETCH.sub(r"\1", text)
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
