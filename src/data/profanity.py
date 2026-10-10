import re

MAT = (
    r"\b(?:бля\w*|бл[яе]т\w*|ху[йеяи]\w*|нахуй\w*|похуй\w*|нихуя\w*|оху[её]\w*|пизд\w*|"
    r"[её]б[аулнои]\w*|въ[её]б\w*|вы[её]б\w*|от[ъь]?[её]б\w*|за[её]б\w*|у[её]б\w*|долб[ао][её]б\w*|"
    r"сука|суки|сук|сучк\w*|пид[оа]р\w*|пидр\w*|муда[кч]\w*|мудил\w*|залуп\w*|г[ао]ндон\w*)\b"
)
_MAT_RX = re.compile(MAT, re.IGNORECASE)


def _hide(match):
    word = match.group()
    if len(word) <= 3:
        return word[0] + "*" * (len(word) - 1)
    return word[:2] + "*" * (len(word) - 3) + word[-1]


def mask(text):
    return _MAT_RX.sub(_hide, str(text))


def mask_deep(value):
    if isinstance(value, str):
        return mask(value)
    if isinstance(value, list):
        return [mask_deep(v) for v in value]
    if isinstance(value, dict):
        return {k: mask_deep(v) for k, v in value.items()}
    return value
