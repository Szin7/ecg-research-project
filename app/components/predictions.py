def predicted_class_and_score(prob: float, threshold: float) -> tuple:
    """Return the thresholded label and the raw positive-class sigmoid score.

    Class-weighted training and threshold selection do not calibrate
    probabilities, so this score must not be described as confidence.
    """
    return ("Arrhythmia" if prob >= threshold else "Normal"), prob
