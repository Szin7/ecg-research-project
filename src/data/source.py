from src.data import mitbih, synthetic


def get_data_source(mode: str):
    if mode == "research":
        return mitbih
    if mode == "demo":
        return synthetic
    raise ValueError(f"unknown data mode: {mode!r}")
