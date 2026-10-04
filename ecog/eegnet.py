"""EEGNet on these trials: a negative result, kept because it is informative.

0.556 against 0.867 for a shrinkage LDA on the same epochs, and below the 0.65-0.75
range EEGNet reaches on four-class EEG in the literature. The limit is 90 trials,
not model capacity, which is the useful thing to say about deep models on hackathon
recordings.

The mean reproduces exactly across machines. The spread across folds does not,
because cuDNN convolution kernels are not deterministic.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _build(torch, nn, F):
    class EEGNet(nn.Module):
        """Lawhern et al. 2018, at the paper's default widths."""

        def __init__(self, n_classes, n_channels, n_samples,
                     F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
            super().__init__()
            self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
            self.bn1 = nn.BatchNorm2d(F1)
            self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout)
            self.sepconv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
            self.bn3 = nn.BatchNorm2d(F2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)
            self.fc = nn.Linear(F2 * (n_samples // 4 // 8), n_classes)

        def forward(self, x):
            x = self.bn1(self.conv1(x))
            x = F.elu(self.bn2(self.depthwise(x)))
            x = self.drop1(self.pool1(x))
            x = F.elu(self.bn3(self.sepconv(x)))
            x = self.drop2(self.pool2(x))
            return self.fc(x.flatten(1))

    return EEGNet


def eegnet_accuracy(epochs, y, n_splits=10, seed=0, n_epochs=300, decimate=4, device=None):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        return {"error": "torch not installed"}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    EEGNet = _build(torch, nn, F)

    X = epochs[..., ::decimate].astype(np.float32)
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-9)
    classes = np.unique(y)
    yi = np.array([{c: i for i, c in enumerate(classes)}[c] for c in y])

    accs = []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, yi):
        torch.manual_seed(seed)
        model = EEGNet(len(classes), X.shape[1], X.shape[2]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.from_numpy(X[tr][:, None]).to(device)
        yt = torch.from_numpy(yi[tr]).long().to(device)
        model.train()
        for _ in range(n_epochs):
            perm = torch.randperm(len(xt), device=device)
            for i in range(0, len(xt), 16):
                b = perm[i : i + 16]
                opt.zero_grad()
                F.cross_entropy(model(xt[b]), yt[b]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X[te][:, None]).to(device)).argmax(1).cpu().numpy()
        accs.append(float((pred == yi[te]).mean()))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "device": device}
