def research_split(cfg: dict) -> dict:
    ds1 = cfg["split"]["ds1_records"]
    val = set(cfg["split"]["val_records"])
    train = [r for r in ds1 if r not in val]
    test = cfg["split"]["ds2_records"]
    return {"train": train, "val": sorted(val), "test": test}


def demo_split(record_ids: list) -> dict:
    """Small, deterministic record-wise split for the synthetic fixture --
    same no-leakage principle as the real split, just without a literature
    partition to follow since these aren't real patients."""
    n = len(record_ids)
    n_val = max(1, n // 6)
    n_test = max(1, n // 6)
    return {
        "test": record_ids[:n_test],
        "val": record_ids[n_test : n_test + n_val],
        "train": record_ids[n_test + n_val :],
    }


def get_split(mode: str, cfg: dict) -> dict:
    if mode == "research":
        return research_split(cfg)
    if mode == "demo":
        from src.data.synthetic import available_records

        return demo_split(available_records())
    raise ValueError(f"unknown mode: {mode!r}")


def assert_no_record_overlap(split: dict) -> None:
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    overlap = (train & val) | (train & test) | (val & test)
    if overlap:
        raise ValueError(f"records appear in more than one split: {sorted(overlap, key=str)}")


def _aami_superclass(symbol: str, cfg: dict):
    """Which of the 5 AAMI EC57 superclasses (N/S/V/F/Q) a raw MIT-BIH
    annotation symbol belongs to, or None for non-beat annotations
    (rhythm markers, artifact) that don't appear in the grouping at all."""
    label_map = cfg["label_map"]
    return next((cls for cls, symbols in label_map.items() if symbol in symbols), None)


def map_symbol_to_label(symbol: str, cfg: dict):
    """Resolve a symbol to the binary task. Returns None for non-beat
    annotations and for classes excluded from the binary task (paced/unknown)."""
    aami_class = _aami_superclass(symbol, cfg)
    if aami_class is None:
        return None
    binary = cfg["binary_task"]
    if aami_class in binary["normal_classes"]:
        return 0
    if aami_class in binary["arrhythmia_classes"]:
        return 1
    return None


AAMI_CLASSES = ["N", "S", "V", "F"]  # Q excluded, same rationale as the binary task


def map_symbol_to_aami_class(symbol: str, cfg: dict):
    """Full 4-class AAMI resolution (N/S/V/F) for the secondary multi-class
    experiment. Q is excluded for the same reason it's excluded from the
    binary task -- paced/unknown beats are a different problem from
    arrhythmia classification, not a class worth asking the model to learn."""
    aami_class = _aami_superclass(symbol, cfg)
    if aami_class is None or aami_class == "Q":
        return None
    return AAMI_CLASSES.index(aami_class)
