"""The number we submit, and the protocol that makes it defensible.

Round two of two. `search.py` found that the window matters and the model does not;
this file turns that into one pipeline with no choices left in it, and then measures
it three ways:

  fixed       three windows concatenated, six bands, shrinkage LDA, 5 x 10 folds.
              Nothing is selected, so there is nothing to select on.
  nested      the choice over all 40 candidate feature sets moved inside the
              cross-validation, 3 x 10 outer folds. This is the honest number, and
              the check no prior work on this recording ran.
  permutation 200 shuffled label vectors against the observed accuracy.

It also re-runs rest versus movement, the one task where our first version lost to
the published MATLAB baseline, with the better window.

    python -m ecog.final --jobs 8      -> results/final.json

`--jobs` changes the wall clock and nothing else: 5 minutes instead of 55. Folds and
permutations are drawn in one sequence and consumed in order, so the output does not
depend on the worker count.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .data import HIGH_GAMMA_3, HIGH_GAMMA_6, cue_onsets, epoch, load, log_band_power, preprocess
from .models import lda, nested_choice

ROOT = Path(__file__).resolve().parents[1]

WINDOWS = [(0.2, 0.7), (0.3, 0.8), (0.5, 1.0), (0.2, 1.0), (0.3, 1.3), (0.5, 1.5), (1.0, 2.0)]
MULTI = {
    "multi3_early": [(0.2, 0.7), (0.5, 1.0), (0.3, 1.3)],
    "multi3_spread": [(0.2, 0.7), (0.5, 1.0), (1.0, 2.0)],
    "multi4": [(0.2, 0.7), (0.3, 0.8), (0.5, 1.0), (0.5, 1.5)],
}
BANDSETS = {
    "hg3": HIGH_GAMMA_3,
    "hg6": HIGH_GAMMA_6,
    "hg_wide": [(70, 170)],
    "broadband": [(8, 13), (13, 30), (30, 60), (60, 90), (110, 140), (160, 190)],
}
CANDIDATES = ("multi3_early/hg6", "multi3_early/hg3", "multi3_spread/hg6", "multi4/hg6")
REST_WINDOWS = ((-2.0, -1.0), (-1.2, -0.2))

def sequential(fn, items):
    return [fn(it) for it in items]


def make_mapper(jobs):
    """A map over independent fits, with one BLAS thread per worker.

    `inner_max_num_threads=1` is the point. Without it each worker opens a thread per
    core, so `--jobs 8` asks a 23-core box for about 80 threads and runs slower than a
    single worker. Wrapping the call in `threadpool_limits` does not help, because that
    does not cross the process boundary.
    """
    if jobs <= 1:
        return sequential

    def run(fn, items):
        if len(items) < 2:
            return sequential(fn, items)
        return Parallel(n_jobs=jobs, backend="loky", inner_max_num_threads=1)(
            delayed(fn)(it) for it in items)

    return run


def feature_bank(clean, onsets, classes):
    """Single windows and multi-window concatenations: 40 candidate feature sets."""
    single, y = {}, None
    for w in WINDOWS:
        ep, keep = epoch(clean, onsets, *w)
        y = classes[keep]
        for name, bands in BANDSETS.items():
            single[(w, name)] = log_band_power(ep, bands)
    bank = {f"{w[0]}-{w[1]}s/{b}": X for (w, b), X in single.items()}
    for multi, windows in MULTI.items():
        for b in BANDSETS:
            bank[f"{multi}/{b}"] = np.concatenate([single[(w, b)] for w in windows], axis=1)
    return bank, np.asarray(y)


def repeated_cv(X, y, n_repeats=5, seed=0, mapper=sequential):
    """Accuracy and the summed confusion matrix over repeated 10-fold splits."""
    labels = np.unique(y)
    splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats, random_state=seed).split(X, y))

    def one_fold(split):
        tr, te = split
        pred = lda().fit(X[tr], y[tr]).predict(X[te])
        return accuracy_score(y[te], pred), confusion_matrix(y[te], pred, labels=labels)

    folds = mapper(one_fold, splits)
    cm = np.zeros((len(labels), len(labels)), int)
    for _, c in folds:
        cm += c
    accs = [a for a, _ in folds]
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "n_folds": len(accs), "confusion_sum": cm.tolist()}


def permutation_p(X, y, observed, n_perm=200, seed=0, mapper=sequential):
    """The permutations are drawn here, in one sequence, so the set tested is fixed."""
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(n_perm)]

    def score(yp):
        accs = [accuracy_score(yp[te], lda().fit(X[tr], yp[tr]).predict(X[te]))
                for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(X, yp)]
        return float(np.mean(accs))

    return (sum(1 for s in mapper(score, perms) if s >= observed) + 1) / (n_perm + 1)


def rest_vs_movement(clean, onsets, bank, log, mapper=sequential):
    """Rest taken from before the cue, movement from the same windows after it."""
    out = {}
    for lo, hi in REST_WINDOWS:
        feats = [log_band_power(epoch(clean, onsets, lo, lo + (b - a))[0], BANDSETS["hg6"])
                 for a, b in MULTI["multi3_early"]]
        X_rest = np.concatenate(feats, axis=1)
        X_move = bank["multi3_early/hg6"]
        n = min(len(X_rest), len(X_move))
        X = np.concatenate([X_rest[:n], X_move[:n]], axis=0)
        y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
        key = f"rest{lo}_{hi}"
        out[key] = repeated_cv(X, y, n_repeats=5, mapper=mapper)
        log(f"rest vs movement {key}: {out[key]['mean']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(ROOT / "results" / "final.json"))
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                    help="worker processes; 1 is the sequential path, same output")
    args = ap.parse_args()

    mapper = make_mapper(max(1, args.jobs))
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
    bank, y = feature_bank(clean, onsets, classes)
    log(f"bank {len(bank)} entries, {len(y)} trials, {args.jobs} workers")

    keys = list(bank)
    means = mapper(lambda k: repeated_cv(bank[k], y, n_repeats=1)["mean"], keys)
    r["sweep_repeated1"] = {k: round(v, 4) for k, v in
                            sorted(zip(keys, means), key=lambda kv: -kv[1])}
    log(f"sweep top: {list(r['sweep_repeated1'].items())[:5]}")
    save()

    for name in CANDIDATES:
        r[f"fixed/{name}"] = repeated_cv(bank[name], y, n_repeats=5, mapper=mapper)
        log(f"fixed {name}: {r[f'fixed/{name}']['mean']:.4f}")
        save()

    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0).split(y, y)
    r["nested_full_bank"] = nested_choice(bank, y, outer=outer, mapper=mapper)
    log(f"nested over the full bank: {r['nested_full_bank']['mean']:.4f}")
    save()

    headline = max(CANDIDATES, key=lambda k: r[f"fixed/{k}"]["mean"])
    observed = repeated_cv(bank[headline], y, n_repeats=1, mapper=mapper)["mean"]
    r["permutation"] = {"feature_set": headline, "observed_single_10fold": observed,
                        "n_permutations": 200,
                        "p_value": permutation_p(bank[headline], y, observed, n_perm=200,
                                                 mapper=mapper)}
    log(f"permutation {r['permutation']}")
    save()

    r["rest_vs_movement"] = rest_vs_movement(clean, onsets, bank, log, mapper=mapper)
    save()
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
