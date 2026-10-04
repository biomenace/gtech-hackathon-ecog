"""Protocols from earlier work on this recording, re-implemented to be comparable.

Each one is scored on the epochs `data.preprocess` produces, so a prior-art number
and ours differ by the method and not by the preprocessing. The runners that call
these live next to the repository they reproduce, under `prior-art/*-helper/`.

Eidantz 2024 classified one gesture against the other two. On balanced 3-class data
that scores 0.667 before any learning, so the majority baseline is returned next to
every accuracy.

neverix 2022 printed up to 0.944, as the maximum over several dozen configurations
read off the same resamples it was measured on. The feature stack here is theirs,
evaluated once per protocol with no maximum taken.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FS, bad_channels, bandpass, cue_onsets, epoch, load, log_band_power, notch_cascade, preprocess
from .models import cv_scores


def load_epochs(path, tmin=1.0, tmax=2.0):
    """The shared entry point for every prior-art runner."""
    ecog, labels, _ = load(path)
    onsets, classes = cue_onsets(labels)
    clean, bad = preprocess(ecog, band=(50.0, 300.0))
    epochs, keep = epoch(clean, onsets, tmin, tmax)
    return epochs, classes[keep], bad


def eidantz_protocol(epochs, y, bad=(), n_splits=10, test_size=0.3, seed=30):
    """Riemannian tangent space, one gesture against the rest, ShuffleSplit(10).

    Their plain sample covariance is singular on these epochs: channels rejected
    before the reference become identical after it. Those rows are dropped, without
    which the tangent-space map fails outright.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    live = np.setdiff1d(np.arange(epochs.shape[1]), np.asarray(bad, dtype=int))
    cov = Covariances(estimator="scm").transform(epochs[:, live].astype(np.float64))
    proto = Pipeline([("ts", TangentSpace()), ("lr", LogisticRegression(max_iter=4000))])

    out, per_class = {}, []
    for gesture in (1, 2, 3):
        yb = (y == gesture).astype(int)
        accs, majority = [], []
        for tr, te in ShuffleSplit(n_splits, test_size=test_size, random_state=seed).split(cov):
            accs.append(accuracy_score(yb[te], clone(proto).fit(cov[tr], yb[tr]).predict(cov[te])))
            majority.append(max(np.mean(yb[te]), 1 - np.mean(yb[te])))
        out[f"gesture_{gesture}_vs_rest"] = {"acc": float(np.mean(accs)),
                                            "majority_baseline": float(np.mean(majority))}
        per_class.append(np.mean(accs))
    out["overall_one_vs_rest"] = float(np.mean(per_class))
    out["note"] = "one-vs-rest on balanced 3-class data: 0.667 is the no-skill score"
    return out


def psd_bins(epochs, low, high, width=10.0, nfft=512, fs=FS):
    """Welch PSD averaged into fixed-width bands.

    The raw per-bin PSD of 60 channels is about 7000 features for 90 trials, which
    makes the Ledoit-Wolf covariance of the LDA the dominant cost of the run without
    adding information at that resolution. Binning keeps the spectral shape.
    """
    from scipy.signal import welch

    freqs, psd = welch(epochs, fs=fs, nperseg=min(nfft, epochs.shape[-1]), axis=-1)
    edges = np.arange(low, high + width, width)
    cols = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (freqs >= lo) & (freqs < hi)
        if inside.any():
            cols.append(np.log(psd[..., inside].mean(axis=-1) + 1e-20))
    return np.stack(cols, axis=-1).reshape(epochs.shape[0], -1)


def neverix_features(epochs):
    """Their feature union: four overlapping bands plus binned Welch PSD.

    CSP is fitted inside the cross-validation loop, so it is added separately.
    """
    feats = []
    for low, high in ((40, 100), (80, 140), (120, 180), (160, 220)):
        f = bandpass(epochs.reshape(-1, epochs.shape[-1]), low, high).reshape(epochs.shape)
        feats.append(np.log(np.mean(f**2, axis=-1) + 1e-12))
    feats.append(psd_bins(epochs, 40, 300, width=10.0))
    feats.append(psd_bins(epochs, 100, 250, width=5.0))
    return np.concatenate(feats, axis=-1)


def _pca_lda():
    """A PCA in front of the LDA, for a reason that is about cost, not accuracy.

    The feature union is about 3600-dimensional for 60 training trials. A Ledoit-Wolf
    LDA at that width costs roughly 115 s per fit, which makes the 48-fold protocol
    infeasible. Sixty training trials span at most 59 dimensions, so PCA(40) discards
    nothing that could have been estimated.
    """
    return Pipeline([("pca", PCA(n_components=40, svd_solver="randomized", random_state=0)),
                     ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


def _neverix_models():
    return (("PCA40+LDA", _pca_lda),
            ("logreg-C0.1", lambda: LogisticRegression(max_iter=4000, C=0.1)))


def _with_csp(epochs64, y, tr, te, Xtr, Xte, components=(4, 8, 12)):
    """Append CSP projections at several component counts, fitted on the fold only."""
    from mne.decoding import CSP

    tr_parts, te_parts = [], []
    for n in components:
        csp = CSP(n_components=n, reg="ledoit_wolf", log=True, norm_trace=False)
        tr_parts.append(csp.fit_transform(epochs64[tr], y[tr]))
        te_parts.append(csp.transform(epochs64[te]))
    return np.concatenate([Xtr] + tr_parts, axis=1), np.concatenate([Xte] + te_parts, axis=1)


def _fold_accuracies(X, y, epochs64, splits, make, use_csp):
    accs = []
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if use_csp:
            Xtr, Xte = _with_csp(epochs64, y, tr, te, Xtr, Xte)
        clf = Pipeline([("scale", StandardScaler()), ("clf", make())]).fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    return accs


def neverix_protocol(epochs, y, seed=16):
    """Their feature stack on their 16 x 3-fold scheme and on our 10-fold one."""
    X = neverix_features(epochs)
    ep64 = epochs.astype(np.float64)
    results = {}

    def score(splits, tag):
        for name, make in _neverix_models():
            for use_csp in (False, True):
                accs = _fold_accuracies(X, y, ep64, splits, make, use_csp)
                key = f"{tag}/{name}{'+CSP' if use_csp else ''}"
                results[key] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
                print(f"  {key}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}", flush=True)

    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(16):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(rng.integers(1 << 30)))
        resampled.extend(list(skf.split(X, y)))
    score(resampled, "16x3fold")
    score(list(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X, y)), "10fold")
    return results


MARIA_BANDS = ((60.0, 90.0), (110.0, 140.0), (160.0, 190.0))
MARIA_WINDOW = (1.0, 2.0)


def maria_reference(ecog, reference):
    """Her preprocessing, with the spatial filter either as written or corrected.

    `reference="broken"` reproduces `y_CAR(i,:) = y_trim(i,:) - med_s(i)` literally:
    `med_s` is the median across channels per sample, so indexing it by the channel
    number subtracts one scalar from the whole channel and no re-referencing happens.
    `reference="fixed"` subtracts the reference signal, which is what the comment in
    her source describes.
    """
    bad = bad_channels(ecog)
    x = ecog.copy()
    if len(bad):
        x[bad] = 0.0

    median = np.median(x, axis=0)
    if reference == "fixed":
        x = x - median[None, :]
    elif reference == "broken":
        x = x - median[: x.shape[0], None]
    else:
        raise ValueError(f"reference must be 'fixed' or 'broken', not {reference!r}")

    return notch_cascade(x), bad


def maria_cue_onsets(labels):
    """Her trial extraction keeps rest as a fourth cue, which is where her 180 comes from."""
    lab = np.rint(labels).astype(int)
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return change, lab[change]


def maria_protocol(ecog, labels, reference, log=print):
    """Her window, her bands, a 100-tree random forest, 10-fold, both of her tasks."""
    from sklearn.ensemble import RandomForestClassifier

    notched, bad = maria_reference(ecog, reference)
    clean = bandpass(notched, 50.0, 300.0)
    onsets, cues = maria_cue_onsets(labels)
    epochs, keep = epoch(clean, onsets, *MARIA_WINDOW)
    cue = cues[keep]
    X = log_band_power(epochs, MARIA_BANDS)
    log(f"{reference} reference: epochs {epochs.shape}, bad channels {bad.tolist()}")

    moving = cue > 0
    out = {}
    for task, (Xt, yt) in (("rest_vs_movement", (X, moving.astype(int))),
                           ("three_gestures", (X[moving], cue[moving]))):
        accs, cm, _ = cv_scores(lambda: RandomForestClassifier(100, random_state=0),
                                Xt, yt, scale=False)
        out[task] = {"acc": round(float(accs.mean()), 4), "std": round(float(accs.std()), 4),
                     "n_trials": int(len(yt)), "n_splits": 10, "confusion": cm.tolist()}
        log(f"  {reference}/{task}: {out[task]['acc']:.3f} +/- {out[task]['std']:.3f} (n={len(yt)})")
    return out, bad
