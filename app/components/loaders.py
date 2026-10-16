import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation.predict import predict_array
from src.models.checkpoint import load_checkpoint
from src.utils.paths import ARTIFACTS_DIR, DATA_DIR

MODEL_TAGS = {"standard": "baseline_cnn", "defended": "defended_cnn"}


def tag_for(mode: str, kind: str) -> str:
    if mode == "research" and (ARTIFACTS_DIR / "active_study.json").exists():
        return {"standard": "v2_context", "defended": "v2_context_defended"}[kind]
    base = MODEL_TAGS[kind]
    return base if mode == "research" else f"demo_{base}"


def _version(path):
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_resource(max_entries=8)
def _model(path, version):
    return load_checkpoint(path)


def get_model(tag):
    path = ARTIFACTS_DIR / "models" / f"{tag}.pt"
    version = _version(path)
    return _model(str(path), version) if version else None


@st.cache_data(max_entries=32)
def _json(path, version):
    return json.loads(Path(path).read_text())


def _read_json(relative):
    path = ARTIFACTS_DIR / relative
    version = _version(path)
    return _json(str(path), version) if version else None


def get_history(tag):
    return _read_json(f"models/{tag}_history.json")


def get_clean_metrics(tag):
    return _read_json(f"experiments/{tag}/clean_metrics.json")


@st.cache_data(max_entries=16)
def _csv(path, version):
    return pd.read_csv(path)


def get_robustness_curve(tag):
    path = ARTIFACTS_DIR / "experiments" / tag / "robustness_curve.csv"
    version = _version(path)
    if version is None:
        return None
    df = _csv(str(path), version)
    df["tag"] = tag
    return df


def get_attack_examples(tag):
    path = ARTIFACTS_DIR / "experiments" / tag / "attack_examples.csv"
    version = _version(path)
    return _csv(str(path), version) if version else None


def model_status(tag):
    return (
        "trained"
        if (ARTIFACTS_DIR / "models" / f"{tag}.pt").exists()
        else "not trained"
    )


@st.cache_data(max_entries=4)
def _split(path, version):
    with np.load(path, allow_pickle=False) as data:
        return data["X"], data["y"], data["groups"]


def test_data_path(mode):
    if mode == "research" and (ARTIFACTS_DIR / "active_study.json").exists():
        full = DATA_DIR / "processed" / "mitbih_subject_disjoint_v2" / "test.npz"
        return full if full.exists() else DATA_DIR / "preview" / "v2_test.npz"
    return DATA_DIR / "processed" / f"{mode}_test.npz"


def get_test_data(mode):
    path = test_data_path(mode)
    if path.parent.name == "preview":
        st.caption(
            "Research preview: 64 fixed beats per record. Saved aggregate clean metrics use all 49,685 test beats."
        )
    version = _version(path)
    if version is None:
        st.info(
            f"Prepare the {mode} dataset to inspect individual beats. Saved benchmark summaries remain available."
        )
        st.code(f"python -m scripts.prepare_data --mode {mode}", language="bash")
        st.stop()
    return _split(str(path), version)


@st.cache_data(max_entries=8)
def _scores(tag, model_version, data_version, _X):
    return predict_array(get_model(tag).model, _X)


def get_test_scores(tag, mode, X):
    return _scores(
        tag,
        _version(ARTIFACTS_DIR / "models" / f"{tag}.pt"),
        _version(test_data_path(mode)),
        X,
    )
