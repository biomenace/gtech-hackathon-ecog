"""The first version of our pipeline: one window, three bands, every classifier.

This is the run that produced 0.867 and, more importantly, the sliding-window curve
that showed why 0.867 was the ceiling. The window here is the published one, 1 s
starting 1 s after the cue, chosen so our rows line up with the 2022 MATLAB result
rather than because it is a good window. It is not a good window, and
`accuracy_over_time` in the same run is what proves it.

    python -m ecog.baseline            -> results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import (FS, GESTURES, HIGH_GAMMA_3, cue_onsets, epoch, load, log_band_power,
                   preprocess, welch_features)
from .eegnet import eegnet_accuracy
from .models import (accuracy_over_time, classical_models, csp_lda, cv_scores, glove_leakage,
                     glove_regression, permutation_p, riemannian)

ROOT = Path(__file__).resolve().parents[1]
WINDOW = (1.0, 2.0)
REST_WINDOW = (-2.0, -1.0)


def gesture_models(X, y, log):
    """Every classical model on identical folds, with its confusion matrix."""
    out = {}
    for name, make in classical_models().items():
        accs, cm, cls = cv_scores(make, X, y)
        out[name] = {"mean": float(accs.mean()), "std": float(accs.std()),
                     "folds": accs.round(4).tolist(), "confusion": cm.tolist(),
                     "labels": [GESTURES[int(c)] for c in cls]}
        log(f"{name}: {accs.mean():.3f} +/- {accs.std():.3f}")
    return out


def rest_vs_movement(clean, onsets, X_move, log):
    """Is there movement at all, as the MATLAB baseline's first task."""
    ep_rest, _ = epoch(clean, onsets, *REST_WINDOW)
    X_rest = log_band_power(ep_rest, HIGH_GAMMA_3)
    X = np.concatenate([X_rest, X_move], axis=0)
    y = np.concatenate([np.zeros(len(X_rest), int), np.ones(len(X_move), int)])
    accs, cm, _ = cv_scores(classical_models()["random-forest"], X, y)
    log(f"rest vs movement: {accs.mean():.3f}")
    return {"mean": float(accs.mean()), "std": float(accs.std()), "confusion": cm.tolist()}


def welch_rows(epochs, y, log):
    """Wide spectral features, the angle the 2022 Python entry took."""
    out = {}
    for low, high, nfft in ((40, 300, 256), (100, 250, 256)):
        Xw = welch_features(epochs, low, high, nfft)
        accs, _, _ = cv_scores(classical_models()["logreg"], Xw, y)
        out[f"{low}-{high}Hz"] = {"mean": float(accs.mean()), "std": float(accs.std()),
                                  "n_features": int(Xw.shape[1])}
        log(f"welch {low}-{high}: {accs.mean():.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline.json"))
    ap.add_argument("--perm", type=int, default=200, help="permutations for the null; 0 skips it")
    ap.add_argument("--eegnet-epochs", type=int, default=300)
    ap.add_argument("--skip-eegnet", action="store_true")
    ap.add_argument("--skip-timecourse", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    ecog, labels, glove = load(args.data)
    onsets, classes = cue_onsets(labels)
    clean, bad = preprocess(ecog)
    log(f"loaded {ecog.shape}, {len(onsets)} trials, bad channels {bad.tolist()}")

    epochs, keep = epoch(clean, onsets, *WINDOW)
    glove_epochs, _ = epoch(glove, onsets, *WINDOW)
    y = classes[keep]
    X = log_band_power(epochs, HIGH_GAMMA_3)

    r = {"data": str(Path(args.data).resolve()), "fs": FS, "n_trials": int(len(onsets)),
         "class_counts": {str(k): int(v) for k, v in zip(*np.unique(classes, return_counts=True))},
         "bad_channels": bad.tolist()}

    r["gesture_3class"] = gesture_models(X, y, log)
    best = max(r["gesture_3class"], key=lambda k: r["gesture_3class"][k]["mean"])
    if args.perm:
        p = permutation_p(classical_models()[best], X, y,
                          r["gesture_3class"][best]["mean"], n_perm=args.perm)
        r["permutation_test"] = {"model": best, "n_perm": args.perm, "p": float(p)}
        log(f"permutation p({best}) = {p:.4f}")

    r["rest_vs_movement"] = rest_vs_movement(clean, onsets, X, log)
    r["welch_psd"] = welch_rows(epochs, y, log)
    r["riemannian"] = riemannian(epochs, y)
    r["csp_lda"] = csp_lda(epochs, y)
    log(f"riemannian {r['riemannian']}, csp {r['csp_lda']}")

    r["glove_baseline_lda"] = glove_leakage(glove_epochs, y)
    r["glove_regression_ridge"] = glove_regression(X, glove_epochs)
    log(f"glove: leakage {r['glove_baseline_lda']['mean']:.3f}, "
        f"regression mean r {r['glove_regression_ridge']['mean_r']:.3f}")

    if not args.skip_eegnet:
        r["eegnet"] = eegnet_accuracy(epochs, y, n_epochs=args.eegnet_epochs)
        log(f"eegnet {r['eegnet']}")

    if not args.skip_timecourse:
        r["accuracy_over_time"] = accuracy_over_time(clean, onsets, classes, HIGH_GAMMA_3)
        r["accuracy_peak"] = max(r["accuracy_over_time"], key=lambda d: d["acc"])
        log(f"peak at t={r['accuracy_peak']['t_center']}s: {r['accuracy_peak']['acc']:.3f}")

    r["runtime_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(r, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
