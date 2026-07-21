import pandas as pd
import pytest

from pipeline.splits import SplitIntegrityError, load_fixed_split, partition_frame_by_id
from pipeline.reproducibility import load_reproducibility_settings


def test_fixed_split_integrity() -> None:
    split = load_fixed_split()
    settings = load_reproducibility_settings()

    assert split.seed == 3102
    assert settings.seed == split.seed
    assert len(split.train_ids) == 4567
    assert len(split.dev_ids) == 1523
    assert len(split.heldout_ids) == 1523
    assert len(split.all_ids) == 7613
    assert len(set(split.all_ids)) == 7613
    assert set(split.train_ids).isdisjoint(split.dev_ids)
    assert set(split.train_ids).isdisjoint(split.heldout_ids)
    assert set(split.dev_ids).isdisjoint(split.heldout_ids)


def test_partition_uses_ids_and_preserves_fixed_order() -> None:
    split = load_fixed_split()
    reverse_ids = list(reversed(split.all_ids))
    shuffled_frame = pd.DataFrame({"id": reverse_ids, "text": ["x"] * len(reverse_ids)})

    partitions = partition_frame_by_id(shuffled_frame, split)

    assert tuple(partitions["train"]["id"]) == split.train_ids
    assert tuple(partitions["dev"]["id"]) == split.dev_ids
    assert tuple(partitions["heldout"]["id"]) == split.heldout_ids


def test_partition_rejects_missing_split_id() -> None:
    split = load_fixed_split()
    incomplete = pd.DataFrame({"id": split.all_ids[:-1]})

    with pytest.raises(SplitIntegrityError, match="missing 1 split IDs"):
        partition_frame_by_id(incomplete, split)
