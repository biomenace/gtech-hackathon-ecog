"""The search: which window, which bands, and four ideas that did not work.

Round one of two. Everything here is scored on the same outer folds as
`baseline.py`, so the rows are comparable, and anything involving a choice is
chosen on inner folds inside the training part. The gap between the 0.978 that the
sweep reports and the 0.95 the nested version reports is exactly the cost of making
that choice honestly.

Four negative results come out of this file, and they are in the README because
they are the answer to "did you try":

  augmented covariance matrices  Carrara & Papadopoulo, IEEE TBME 71(9) 2024.
      Delay-embed the trial before taking the covariance, so the matrix carries
      temporal structure. Beats plain Riemannian methods on MOABB. Here it loses,
      because the gesture lives in the power envelope and not in phase.
  sub-window voting              Each trial contributes several shifted windows as
      training examples and is decided by averaging over its own. Works, but no
      better than concatenating windows, which is simpler.
  soft voting over five views    Loses to its own best member.
  trees and boosting             Lose to the linear model.

    python -m ecog.search            -> results/search.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .data import FS, HIGH_GAMMA_3, HIGH_GAMMA_6, cue_onsets, epoch, load, log_band_power, preprocess
from .models import OUTER, lda, logreg, nested_choice, plain_cv

ROOT = Path(__file__).resolve().parents[1]

WINDOWS = [(0.2, 0.7), (0.3, 0.8), (0.5, 1.0), (0.2, 1.0), (0.3, 1.3), (0.5, 1.5), (1.0, 2.0)]
BANDSETS = {
    "hg3": HIGH_GAMMA_3,
    "hg6": HIGH_GAMMA_6,
    "hg_wide": [(70, 170)],
    "broadband": [(8, 13), (13, 30), (30, 60), (60, 90), (110, 140), (160, 190)],
}
ACM_WINDOW = (0.2, 1.0)


def feature_bank(clean, onsets, classes):
    """Log band power for every (window, band set) pair. Returns (bank, epochs, y)."""
    bank, epochs, y = {}, {}, None
    for w in WINDOWS:
        ep, keep = epoch(clean, onsets, *w)
        epochs[w], y = ep, classes[keep]
        for name, bands in BANDSETS.items():
            bank[f"{w[0]}-{w[1]}s/{name}"] = log_band_power(ep, bands)
    return bank, epochs, np.asarray(y)


def delay_embedded_cov(epochs, order, lag, estimator="oas"):
    """Stack `order` copies of the trial shifted by multiples of `lag`, then covary.

    order=1 is the ordinary spatial covariance. Higher orders add the lagged
    cross-covariances between channels, which is the whole idea of the method.
    """
    from pyriemann.estimation import Covariances

    n = epochs.shape[-1] - (order - 1) * lag
    aug = np.concatenate([epochs[..., i * lag : i * lag + n] for i in range(order)], axis=1)
    return Covariances(estimator=estimator).transform(aug.astype(np.float64))


def strongest_channels(epochs, k):
    """Rank channels by how much their log power varies across trials.

    Labels are never read, so this is computed once outside the folds without leaking.
    """
    power = np.log(np.mean(epochs**2, axis=-1) + 1e-12)
    return np.argsort(power.var(axis=0))[::-1][:k]


def tangent_space():
    from pyriemann.tangentspace import TangentSpace

    return Pipeline([("ts", TangentSpace()), ("lr", LogisticRegression(max_iter=4000, C=1.0))])


def augmented_covariance(epochs, y, log, orders=(1, 2, 3), lags=(8, 16, 32), n_ch=24):
    from pyriemann.classification import FgMDM, MDM

    sel = strongest_channels(epochs, n_ch)
    sub = epochs[:, sel]
    bank = {f"order{o}_lag{l}": delay_embedded_cov(sub, o, l)
            for o in orders for l in lags if not (o == 1 and l != lags[0])}

    out = {"n_channels": int(n_ch), "channels": sel.tolist(), "plain": {}}
    for name, cov in bank.items():
        key = f"{name}/TS+logreg"
        out["plain"][key] = plain_cv(tangent_space, cov, y)
        log(f"  ACM {name}: {out['plain'][key]['mean']:.3f}")
    out["nested"] = nested_choice(bank, y, make_clf=tangent_space)
    log(f"  ACM nested: {out['nested']['mean']:.3f}")
    out["MDM_order2_lag16"] = plain_cv(MDM, bank["order2_lag16"], y)
    out["FgMDM_order2_lag16"] = plain_cv(FgMDM, bank["order2_lag16"], y)
    return out


def subwindow_vote(clean, onsets, y, bands, start=0.1, stop=1.3, width=0.4, step=0.15, fs=FS):
    """Train on shifted sub-windows, decide a trial by averaging over its own.

    The cheapest way to buy effective sample size at 90 trials. The trial is still
    decided once, so the fold structure is untouched.
    """
    starts = np.arange(start, stop - width + 1e-9, step)
    X = np.stack([log_band_power(epoch(clean, onsets, s, s + width, fs=fs)[0], bands) for s in starts], axis=1)
    accs = []
    for tr, te in StratifiedKFold(**OUTER).split(X, y):
        m = lda().fit(X[tr].reshape(-1, X.shape[-1]), np.repeat(y[tr], X.shape[1]))
        df = m.decision_function(X[te].reshape(-1, X.shape[-1])).reshape(len(te), X.shape[1], -1).mean(axis=1)
        accs.append(accuracy_score(y[te], m.classes_[df.argmax(axis=1)]))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "n_subwindows": int(X.shape[1]), "width_s": width, "step_s": step,
            "span_s": [float(start), float(stop)]}


def soft_vote(views, y):
    """Average the class probabilities of one classifier per feature view."""
    names = list(views)
    accs, per_view = [], {n: [] for n in names}
    for tr, te in StratifiedKFold(**OUTER).split(y, y):
        proba, classes = None, None
        for n in names:
            X, make = views[n]
            m = make().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            per_view[n].append(accuracy_score(y[te], m.classes_[p.argmax(axis=1)]))
            proba = p if proba is None else proba + p
            classes = m.classes_
        accs.append(accuracy_score(y[te], classes[proba.argmax(axis=1)]))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "per_view": {n: round(float(np.mean(v)), 4) for n, v in per_view.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(ROOT / "results" / "search.json"))
    args = ap.parse_args()

    t0 = time.time()
    r = {}

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    def save():
        r["runtime_s"] = round(time.time() - t0, 1)
        Path(args.out).write_text(json.dumps(r, indent=2))

    ecog, labels, _ = load(args.data)
    onsets, classes = cue_onsets(labels)
    clean, bad = preprocess(ecog)
    r["bad_channels"] = bad.tolist()
    bank, epochs, y = feature_bank(clean, onsets, classes)
    log(f"bad channels {bad.tolist()}, feature bank {len(bank)} entries")

    sweep = {k: plain_cv(lda, X, y) for k, X in bank.items()}
    best = max(sweep, key=lambda k: sweep[k]["mean"])
    r["sweep"] = {k: round(v["mean"], 4) for k, v in sweep.items()}
    r["sweep_best"] = {"key": best, **sweep[best]}
    log(f"best of sweep {best} {sweep[best]['mean']:.3f}")
    save()

    r["nested_window_band"] = nested_choice(bank, y)
    log(f"nested over the sweep: {r['nested_window_band']['mean']:.3f}")
    save()

    r["multi_window_concat"] = plain_cv(
        lda, np.concatenate([bank[f"{a}-{b}s/hg3"] for a, b in [(0.2, 0.7), (0.5, 1.0), (1.0, 2.0)]], axis=1), y)
    log(f"three windows concatenated: {r['multi_window_concat']['mean']:.3f}")
    save()

    r["subwindow_vote"] = subwindow_vote(clean, onsets, y, BANDSETS["hg3"])
    log(f"sub-window voting: {r['subwindow_vote']['mean']:.3f}")
    save()

    log(f"augmented covariance on window {ACM_WINDOW[0]}-{ACM_WINDOW[1]} s")
    r["acm"] = augmented_covariance(epochs[ACM_WINDOW], y, log)
    save()

    X_hg = bank[f"{ACM_WINDOW[0]}-{ACM_WINDOW[1]}s/hg3"]
    r["trees"] = {
        "random_forest": plain_cv(lambda: RandomForestClassifier(500, random_state=0), X_hg, y),
        "hist_gradient_boosting": plain_cv(
            lambda: HistGradientBoostingClassifier(max_iter=300, random_state=0), X_hg, y),
    }
    log(f"trees {r['trees']}")
    save()

    sel = strongest_channels(epochs[ACM_WINDOW], 24)
    r["ensemble"] = soft_vote({
        "hg3_lda": (X_hg, lda),
        "hg6_lda": (bank[f"{ACM_WINDOW[0]}-{ACM_WINDOW[1]}s/hg6"], lda),
        "broadband_logreg": (bank[f"{ACM_WINDOW[0]}-{ACM_WINDOW[1]}s/broadband"], logreg),
        "acm_ts": (delay_embedded_cov(epochs[ACM_WINDOW][:, sel], 2, 16), tangent_space),
        "rf": (X_hg, lambda: RandomForestClassifier(500, random_state=0)),
    }, y)
    log(f"soft vote {r['ensemble']['mean']:.3f}, best member "
        f"{max(r['ensemble']['per_view'].values()):.3f}")
    save()
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
