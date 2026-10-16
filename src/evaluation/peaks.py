import numpy as np


def peak_agreement(reference, detected, fs, tolerance_ms=50):
    """One-to-one matches within a stated timing tolerance."""
    reference = np.sort(reference)
    detected = np.sort(detected)
    tolerance = fs * tolerance_ms / 1000
    i = j = 0
    errors = []
    while i < len(reference) and j < len(detected):
        difference = detected[j] - reference[i]
        if abs(difference) <= tolerance:
            errors.append(abs(difference) * 1000 / fs)
            i += 1
            j += 1
        elif difference < 0:
            j += 1
        else:
            i += 1
    return {
        "matched": len(errors),
        "precision": len(errors) / len(detected) if len(detected) else 0.0,
        "recall": len(errors) / len(reference) if len(reference) else 0.0,
        "mean_absolute_error_ms": float(np.mean(errors)) if errors else None,
    }
