"""Pure, deterministic text normalization levers."""
import re
import unicodedata
from dataclasses import dataclass

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1+")


def is_emoji(ch: str) -> bool:
    return bool(ch) and (ord(ch) >= 0x1F000 or unicodedata.category(ch) == "So")


@dataclass(frozen=True)
class NormalizationConfig:
    url: str = "preserve"
    mention: str = "preserve"
    hashtag: str = "preserve"
    punctuation: str = "preserve"
    casing: str = "lower"
    emoji: str = "preserve"


def normalize_text(value: object, cfg: NormalizationConfig = NormalizationConfig()) -> str:
    text = "" if value is None else str(value)
    if text.lower() == "nan":
        text = ""
    if cfg.url == "replace": text = URL_RE.sub(" URLTOKEN ", text)
    elif cfg.url == "remove": text = URL_RE.sub(" ", text)
    if cfg.mention == "replace": text = MENTION_RE.sub(" USERTOKEN ", text)
    elif cfg.mention == "remove": text = MENTION_RE.sub(" ", text)
    if cfg.hashtag == "strip": text = HASHTAG_RE.sub(r"\1", text)
    elif cfg.hashtag == "remove": text = HASHTAG_RE.sub(" ", text)
    if cfg.punctuation == "remove": text = PUNCT_RE.sub(" ", text)
    elif cfg.punctuation == "repeat": text = REPEATED_PUNCT_RE.sub(r"\1", text)
    if cfg.emoji == "remove": text = "".join(" " if is_emoji(c) else c for c in text)
    elif cfg.emoji == "replace": text = "".join(" EMOJITOKEN " if is_emoji(c) else c for c in text)
    if cfg.casing == "lower": text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


NORMALIZATION_CANDIDATES = {
    "raw": NormalizationConfig(),
    "url_replace": NormalizationConfig(url="replace"),
    "url_remove": NormalizationConfig(url="remove"),
    "mention_replace": NormalizationConfig(mention="replace"),
    "mention_remove": NormalizationConfig(mention="remove"),
    "hashtag_strip": NormalizationConfig(hashtag="strip"),
    "hashtag_remove": NormalizationConfig(hashtag="remove"),
    "preserve_case": NormalizationConfig(casing="preserve"),
    "punct_remove": NormalizationConfig(punctuation="remove"),
    "punct_repeat": NormalizationConfig(punctuation="repeat"),
    "emoji_remove": NormalizationConfig(emoji="remove"),
    "emoji_replace": NormalizationConfig(emoji="replace"),
}

