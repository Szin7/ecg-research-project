import numpy as np


def subject_bootstrap(y, predictions, groups, repetitions=2000, seed=42):
    """95% percentile intervals from resampling whole test subjects."""
    counts = []
    for subject in np.unique(groups):
        keep = groups == subject
        counts.append(
            [
                np.sum(keep & (y == 0) & (predictions == 0)),
                np.sum(keep & (y == 0) & (predictions == 1)),
                np.sum(keep & (y == 1) & (predictions == 0)),
                np.sum(keep & (y == 1) & (predictions == 1)),
            ]
        )
    counts = np.asarray(counts)
    rng = np.random.default_rng(seed)
    draws = counts[rng.integers(0, len(counts), (repetitions, len(counts)))].sum(axis=1)
    tn, fp, fn, tp = draws.T

    def ratio(a, b):
        return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=b > 0)

    values = {
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "recall": ratio(tp, tp + fn),
        "precision": ratio(tp, tp + fp),
        "accuracy": ratio(tp + tn, draws.sum(axis=1)),
    }
    return {
        key: {
            "lower": float(np.nanquantile(v, 0.025)),
            "upper": float(np.nanquantile(v, 0.975)),
            "defined_resamples": int(np.isfinite(v).sum()),
        }
        for key, v in values.items()
    }
