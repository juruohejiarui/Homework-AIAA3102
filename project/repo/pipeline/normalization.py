"""Composable, independently switchable text-normalization transformations."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_MARKER_PATTERN = re.compile(r"#(?=\w)")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # regional indicators
    "\U0001F300-\U0001F5FF"  # pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport and map
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"          # miscellaneous symbols
    "\u2700-\u27BF"          # dingbats
    "]+(?:\uFE0F|\u200D[\U0001F300-\U0001FAFF\u2600-\u27BF]+)*"
)

URL_TOKEN = " URLTOKEN "
MENTION_TOKEN = " MENTIONTOKEN "
EMOJI_TOKEN = " EMOJITOKEN "

NORMALIZATION_NAMES = (
    "raw_text_control",
    "normalize_urls_placeholder",
    "normalize_mentions_placeholder",
    "strip_hashtag_marker",
    "normalize_punctuation_to_space",
    "unicode_casefold",
    "normalize_emoji_placeholder",
)

NORMALIZATION_PARAMETERS: dict[str, dict[str, bool]] = {
    "raw_text_control": {},
    "normalize_urls_placeholder": {"replace_urls": True},
    "normalize_mentions_placeholder": {"replace_mentions": True},
    "strip_hashtag_marker": {"strip_hashtag_markers": True},
    "normalize_punctuation_to_space": {"punctuation_to_space": True},
    "unicode_casefold": {"casefold_text": True},
    "normalize_emoji_placeholder": {"replace_emoji": True},
}


def _punctuation_to_space(text: str) -> str:
    return "".join(" " if unicodedata.category(character).startswith("P") else character for character in text)


def normalize_text(
    text: str,
    *,
    replace_urls: bool = False,
    replace_mentions: bool = False,
    strip_hashtag_markers: bool = False,
    punctuation_to_space: bool = False,
    casefold_text: bool = False,
    replace_emoji: bool = False,
) -> str:
    """Apply only the explicitly enabled transformations in a fixed order."""

    if not isinstance(text, str):
        raise TypeError("text normalization requires string inputs")
    normalized = text
    if replace_urls:
        normalized = URL_PATTERN.sub(URL_TOKEN, normalized)
    if replace_mentions:
        normalized = MENTION_PATTERN.sub(MENTION_TOKEN, normalized)
    if strip_hashtag_markers:
        normalized = HASHTAG_MARKER_PATTERN.sub("", normalized)
    if replace_emoji:
        normalized = EMOJI_PATTERN.sub(EMOJI_TOKEN, normalized)
    if punctuation_to_space:
        normalized = _punctuation_to_space(normalized)
    if casefold_text:
        normalized = normalized.casefold()
    return normalized


class TextNormalizer(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer exposing each normalization as a boolean switch."""

    def __init__(
        self,
        *,
        replace_urls: bool = False,
        replace_mentions: bool = False,
        strip_hashtag_markers: bool = False,
        punctuation_to_space: bool = False,
        casefold_text: bool = False,
        replace_emoji: bool = False,
    ) -> None:
        self.replace_urls = replace_urls
        self.replace_mentions = replace_mentions
        self.strip_hashtag_markers = strip_hashtag_markers
        self.punctuation_to_space = punctuation_to_space
        self.casefold_text = casefold_text
        self.replace_emoji = replace_emoji

    def fit(self, texts: Iterable[str], y: object = None) -> "TextNormalizer":
        del y
        # Consume nothing: this transformer has no learned state.
        return self

    def transform(self, texts: Iterable[str]) -> list[str]:
        return [
            normalize_text(
                text,
                replace_urls=self.replace_urls,
                replace_mentions=self.replace_mentions,
                strip_hashtag_markers=self.strip_hashtag_markers,
                punctuation_to_space=self.punctuation_to_space,
                casefold_text=self.casefold_text,
                replace_emoji=self.replace_emoji,
            )
            for text in texts
        ]


def make_normalizer(name: str) -> TextNormalizer:
    try:
        parameters = NORMALIZATION_PARAMETERS[name]
    except KeyError as error:
        raise ValueError(f"unknown normalization variant {name!r}") from error
    return TextNormalizer(**parameters)


def perturb_urls(text: str) -> str:
    return URL_PATTERN.sub("https://surface-change.invalid/path", text)


def perturb_mentions(text: str) -> str:
    return MENTION_PATTERN.sub("@surface_change_user", text)


def perturb_hashtags(text: str) -> str:
    return HASHTAG_MARKER_PATTERN.sub("", text)


def perturb_punctuation(text: str) -> str:
    return "".join("!" if unicodedata.category(character).startswith("P") else character for character in text)


def perturb_casing(text: str) -> str:
    return text.swapcase()


def perturb_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("🌀", text)


PERTURBATIONS = {
    "normalize_urls_placeholder": perturb_urls,
    "normalize_mentions_placeholder": perturb_mentions,
    "strip_hashtag_marker": perturb_hashtags,
    "normalize_punctuation_to_space": perturb_punctuation,
    "unicode_casefold": perturb_casing,
    "normalize_emoji_placeholder": perturb_emoji,
}
