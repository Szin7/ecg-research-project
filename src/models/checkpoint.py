from dataclasses import dataclass

import torch

from src.models.cnn1d import ECGCNN


def build_model(model_cfg: dict) -> ECGCNN:
    return ECGCNN(
        in_channels=model_cfg["in_channels"],
        conv_channels=tuple(model_cfg["conv_channels"]),
        kernel_sizes=tuple(model_cfg["kernel_sizes"]),
        fc_hidden=model_cfg["fc_hidden"],
        dropout=model_cfg["dropout"],
        num_classes=model_cfg.get("num_classes", 1),
        rr_features=model_cfg.get("rr_features", 0),
        waveform_samples=model_cfg.get("waveform_samples"),
    )


@dataclass
class LoadedModel:
    model: ECGCNN
    threshold: float


def load_checkpoint(path) -> LoadedModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # older checkpoints saved before threshold calibration default to 0.5
    return LoadedModel(model=model, threshold=ckpt.get("threshold", 0.5))
