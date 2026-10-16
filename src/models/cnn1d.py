import torch
import torch.nn as nn


class ECGConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        # Default momentum=0.1 takes ~40+ batches to pull running stats away
        # from their init (mean=0, var=1) toward the true activation scale.
        # Stacked across 3 BN layers that lag matters a lot at eval time,
        # especially early in training or on a small dataset with few
        # batches/epoch -- raised to converge in a handful of epochs instead.
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.2)
        self.pool = nn.MaxPool1d(2) if pool else None

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        return self.pool(x) if self.pool is not None else x


class ECGCNN(nn.Module):
    """Conv1d/BN/ReLU/[Pool] x3 -> global average pool -> dense -> logits.

    The last conv block skips pooling so global average pooling still has
    a few timesteps of context to average over instead of a single point.
    num_classes=1 (default, the binary task) outputs a single raw logit
    for BCEWithLogitsLoss; num_classes>1 (the secondary multi-class
    experiment) outputs one logit per class for CrossEntropyLoss.
    """

    def __init__(
        self,
        in_channels=1,
        conv_channels=(16, 32, 64),
        kernel_sizes=(7, 5, 3),
        fc_hidden=32,
        dropout=0.3,
        num_classes=1,
        rr_features=0,
        waveform_samples=None,
    ):
        super().__init__()
        self.rr_features = rr_features
        self.waveform_samples = waveform_samples
        channels = [in_channels] + list(conv_channels)
        self.blocks = nn.ModuleList(
            [
                ECGConvBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_sizes[i],
                    pool=(i < len(conv_channels) - 1),
                )
                for i in range(len(conv_channels))
            ]
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(conv_channels[-1] + rr_features, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        context = None
        if self.rr_features:
            if x.shape[-1] != self.waveform_samples + self.rr_features:
                raise ValueError("Context model needs waveform and RR features")
            context = x[:, 0, self.waveform_samples :]
        if self.waveform_samples:
            x = x[..., : self.waveform_samples]
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        if context is not None:
            x = torch.cat([x, context], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        # squeeze(-1) is a no-op when num_classes > 1 (only removes a dim
        # of size 1), so this returns (batch,) for binary and (batch,
        # num_classes) for multi-class without branching on which task it is
        return logits.squeeze(-1)

    def attack_mask(self, x):
        mask = torch.ones_like(x)
        if self.waveform_samples:
            mask[..., self.waveform_samples :] = 0
        return mask
