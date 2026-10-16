import numpy as np
import pytest

from src.data.dataset import build_beats, class_distribution, compute_class_weights, compute_pos_weight
from src.data.splits import (
    AAMI_CLASSES,
    assert_no_record_overlap,
    demo_split,
    get_split,
    map_symbol_to_aami_class,
    map_symbol_to_label,
    research_split,
)
from src.utils.config import load_config


@pytest.fixture
def cfg():
    return load_config("dataset.yaml")


def test_research_split_has_no_record_overlap(cfg):
    assert_no_record_overlap(research_split(cfg))


def test_research_split_val_is_subset_of_ds1(cfg):
    split = research_split(cfg)
    ds1 = set(cfg["split"]["ds1_records"])
    assert set(split["val"]) <= ds1
    assert set(split["train"]) <= ds1
    assert set(split["train"]).isdisjoint(split["val"])


def test_demo_split_has_no_overlap_and_covers_all_records():
    record_ids = [f"demo{i:02d}" for i in range(6)]
    split = demo_split(record_ids)
    assert_no_record_overlap(split)
    covered = set(split["train"]) | set(split["val"]) | set(split["test"])
    assert covered == set(record_ids)


def test_assert_no_record_overlap_detects_a_leak():
    bad_split = {"train": [1, 2, 3], "val": [3, 4], "test": [5]}
    with pytest.raises(ValueError):
        assert_no_record_overlap(bad_split)


def test_get_split_rejects_unknown_mode(cfg):
    with pytest.raises(ValueError):
        get_split("bogus", cfg)


def test_map_symbol_to_label_normal_and_arrhythmia(cfg):
    assert map_symbol_to_label("N", cfg) == 0
    assert map_symbol_to_label("L", cfg) == 0
    assert map_symbol_to_label("V", cfg) == 1
    assert map_symbol_to_label("A", cfg) == 1


def test_map_symbol_to_label_excludes_paced_and_unknown(cfg):
    assert map_symbol_to_label("Q", cfg) is None
    assert map_symbol_to_label("/", cfg) is None


def test_map_symbol_to_label_drops_non_beat_annotations(cfg):
    assert map_symbol_to_label("+", cfg) is None


def test_class_distribution_and_pos_weight():
    y = np.array([0, 0, 0, 0, 1, 1])
    assert class_distribution(y) == {0: 4, 1: 2}
    assert compute_pos_weight(y) == pytest.approx(2.0)


def test_compute_pos_weight_handles_no_positives():
    assert compute_pos_weight(np.array([0, 0, 0])) == 1.0


def test_build_beats_on_demo_split_produces_consistent_shapes(cfg):
    split = get_split("demo", cfg)
    X, y, groups = build_beats("demo", split["train"], cfg)

    expected_len = cfg["beat_window"]["pre_r_samples"] + cfg["beat_window"]["post_r_samples"]
    assert X.shape[1] == expected_len
    assert X.shape[0] == len(y) == len(groups)
    assert not np.isnan(X).any()
    assert set(groups.tolist()) <= set(split["train"])


def test_build_beats_train_and_test_records_dont_overlap(cfg):
    split = get_split("demo", cfg)
    _, _, train_groups = build_beats("demo", split["train"], cfg)
    _, _, test_groups = build_beats("demo", split["test"], cfg)

    assert set(train_groups.tolist()).isdisjoint(set(test_groups.tolist()))


def test_map_symbol_to_aami_class_resolves_all_four_classes(cfg):
    assert map_symbol_to_aami_class("N", cfg) == AAMI_CLASSES.index("N")
    assert map_symbol_to_aami_class("A", cfg) == AAMI_CLASSES.index("S")
    assert map_symbol_to_aami_class("V", cfg) == AAMI_CLASSES.index("V")
    assert map_symbol_to_aami_class("F", cfg) == AAMI_CLASSES.index("F")


def test_map_symbol_to_aami_class_excludes_q_and_non_beats(cfg):
    assert map_symbol_to_aami_class("Q", cfg) is None
    assert map_symbol_to_aami_class("/", cfg) is None
    assert map_symbol_to_aami_class("+", cfg) is None


def test_build_beats_with_aami_label_fn_produces_four_classes(cfg):
    split = get_split("demo", cfg)
    X, y, groups = build_beats("demo", split["train"], cfg, label_fn=map_symbol_to_aami_class)

    assert X.shape[0] == len(y) == len(groups)
    assert set(y.tolist()) <= {0, 1, 2, 3}


def test_compute_class_weights_favors_rare_classes():
    y = np.array([0] * 90 + [1] * 6 + [2] * 3 + [3] * 1)
    weights = compute_class_weights(y, num_classes=4)

    assert len(weights) == 4
    assert weights[3] > weights[2] > weights[1] > weights[0]


def test_compute_class_weights_handles_a_missing_class():
    y = np.array([0, 0, 1, 1])  # class 2 never appears
    weights = compute_class_weights(y, num_classes=3)
    assert np.isfinite(weights).all()
