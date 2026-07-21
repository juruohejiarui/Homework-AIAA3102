import pytest

from pipeline.normalization import (
    NORMALIZATION_PARAMETERS,
    PERTURBATIONS,
    make_normalizer,
    normalize_text,
)


@pytest.mark.parametrize(
    ("variant", "text", "expected_fragment", "removed_fragment"),
    [
        ("normalize_urls_placeholder", "see https://x.test/a?q=1 now", "URLTOKEN", "x.test"),
        ("normalize_mentions_placeholder", "hello @Some_User today", "MENTIONTOKEN", "Some_User"),
        ("strip_hashtag_marker", "urgent #FloodAlert", "FloodAlert", "#"),
        ("normalize_punctuation_to_space", "can't-stop_now", "can t stop now", "-"),
        ("unicode_casefold", "STRASSE Straße", "strasse strasse", "STRASSE"),
        ("normalize_emoji_placeholder", "warning 🔥🔥 now", "EMOJITOKEN", "🔥"),
    ],
)
def test_each_normalization_is_independently_switchable(
    variant: str,
    text: str,
    expected_fragment: str,
    removed_fragment: str,
) -> None:
    transformed = make_normalizer(variant).transform([text])[0]

    assert expected_fragment in transformed
    assert removed_fragment not in transformed


def test_raw_control_is_identity() -> None:
    text = "@User #Alert: Straße 🔥 https://example.test/a"

    assert make_normalizer("raw_text_control").transform([text]) == [text]


def test_transformations_are_composable_without_implicit_switches() -> None:
    text = "@User #Alert! 🔥 https://example.test/a"

    transformed = normalize_text(text, replace_urls=True, replace_mentions=True)

    assert "URLTOKEN" in transformed
    assert "MENTIONTOKEN" in transformed
    assert "#Alert!" in transformed
    assert "🔥" in transformed


@pytest.mark.parametrize("variant", list(PERTURBATIONS))
def test_matching_normalizer_is_invariant_to_its_surface_perturbation(variant: str) -> None:
    samples = {
        "normalize_urls_placeholder": "see https://one.test/path now",
        "normalize_mentions_placeholder": "hello @first_user now",
        "strip_hashtag_marker": "urgent #FloodAlert now",
        "normalize_punctuation_to_space": "can't-stop_now!",
        "unicode_casefold": "Mixed CASE Straße",
        "normalize_emoji_placeholder": "fire 🔥 warning",
    }
    original = samples[variant]
    perturbed = PERTURBATIONS[variant](original)
    normalizer = make_normalizer(variant)

    assert perturbed != original
    assert normalizer.transform([original]) == normalizer.transform([perturbed])


def test_variant_registry_changes_at_most_one_switch() -> None:
    for name, parameters in NORMALIZATION_PARAMETERS.items():
        enabled = [key for key, value in parameters.items() if value]
        expected = 0 if name == "raw_text_control" else 1
        assert len(enabled) == expected
