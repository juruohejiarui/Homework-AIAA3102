import numpy as np
import pandas as pd

from pipeline.reproducibility import load_reproducibility_settings
from pipeline.shortcut_features import (
    MISSING_KEYWORD,
    MISSING_LOCATION,
    LengthFeatureTransformer,
    ShallowFeatureTransformer,
    VARIANT_COMPONENTS,
    make_shortcut_pipeline,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["Short #Alert! https://x.test/a", "LONGER report @user 123"],
            "keyword": [np.nan, "flood"],
            "location": ["Paris", np.nan],
            "target": [0, 1],
        }
    )


def test_length_and_shallow_features_are_finite_and_named() -> None:
    frame = _frame()
    length = LengthFeatureTransformer().fit_transform(frame[["text"]])
    shallow = ShallowFeatureTransformer().fit_transform(
        frame[["text", "keyword", "location"]]
    )

    assert length.shape == (2, 3)
    assert shallow.shape == (2, 11)
    assert np.isfinite(length).all()
    assert np.isfinite(shallow).all()
    assert shallow[0, -2] == 1
    assert shallow[1, -1] == 1


def test_missing_metadata_is_explicitly_imputed_inside_pipeline() -> None:
    frame = _frame()
    model = make_shortcut_pipeline("keyword_plus_location", load_reproducibility_settings())
    transformed = model.named_steps["features"].fit_transform(frame, frame["target"])
    names = model.named_steps["features"].get_feature_names_out().tolist()

    assert transformed.shape[0] == 2
    assert any(MISSING_KEYWORD in name for name in names)
    assert any(MISSING_LOCATION in name for name in names)


def test_declared_variants_use_only_known_components() -> None:
    allowed = {"text", "keyword", "location", "length", "shallow"}
    for components in VARIANT_COMPONENTS.values():
        assert set(components) <= allowed
        assert len(components) == len(set(components))
