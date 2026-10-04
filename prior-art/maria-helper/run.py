#!/usr/bin/env python3
"""Measure the MariaGoniIba 2022 protocol, ported from her MATLAB.

Her code needs the Signal Processing and Statistics toolboxes, which we do not have,
and Octave has no `TreeBagger`. Rather than quote her figure and leave it
unverifiable, `ecog.prior.maria_protocol` ports the protocol: her window, her three
bands, log band power per channel, a 100-tree random forest, 10-fold, both of her
tasks. It reproduces her trial counts exactly, 180 and 90.

It is a port and not her binary. `TreeBagger` and `RandomForestClassifier` are both
bagged CART ensembles with different defaults, and her fold assignment is unseeded,
so a point or two of difference from her published figure is expected.

The run also measures her spatial-filter bug both ways, which is the only way to say
what the bug is worth rather than asserting it.

    python prior-art/maria-helper/run.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ecog.data import load
from ecog.prior import MARIA_BANDS, MARIA_WINDOW, maria_protocol

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUBMODULE = HERE / "BR41N.IO-Hackathon-ECoG-Epilepsy"

PUBLISHED = {
    "source": "Results.png in the upstream repository",
    "three_gestures": {"accuracy": 0.8111, "printed_as": "73", "n_trials": 90,
                       "balanced_accuracy": 0.79286, "f1": 0.8},
    "rest_vs_movement": {"accuracy": 0.7889, "printed_as": "142", "n_trials": 180,
                         "balanced_accuracy": 0.78889, "f1": 0.76543},
    "note": "Results.Acc in RF.m is a count of correct trials, not a rate; divided here by 90 and 180",
}
BUGS = {
    "spatial_filter": "preprocessing.m subtracts med_s(i), one scalar per channel, instead of the "
                      "reference signal med_s. Measured as the broken/fixed pair above.",
    "accuracy_as_count": "RF.m reports a count, not a rate. The upstream author marked that line "
                         "'%corregir!' themselves.",
    "patch": "car_fix.patch",
}


def patch_still_applies():
    """Does car_fix.patch still apply to the pinned commit? None if not checked out.

    The patch is the evidence for the two bugs the README describes, so a patch that
    stopped applying would mean the README describes code that is not there.
    """
    if not (SUBMODULE / "preprocessing.m").is_file():
        return None
    r = subprocess.run(["git", "-C", str(SUBMODULE), "apply", "--check", str(HERE / "car_fix.patch")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"car_fix.patch does NOT apply to the pinned commit: {r.stderr.strip()}", file=sys.stderr)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(HERE / "metrics.json"))
    args = ap.parse_args()

    t0 = time.time()
    applies = patch_still_applies()
    print("car_fix.patch applies to the pinned commit: "
          f"{'yes' if applies else 'not cloned yet' if applies is None else 'NO'}", flush=True)

    ecog, labels, _ = load(args.data)
    measured = {}
    for reference in ("broken", "fixed"):
        measured[reference], bad = maria_protocol(ecog, labels, reference)

    Path(args.out).write_text(json.dumps({
        "upstream": "https://github.com/MariaGoniIba/BR41N.IO-Hackathon-ECoG-Epilepsy",
        "commit": "e2e61599edf80693f2a4cd027771ecc4f7da129a",
        "year": 2022,
        "dataset": "ECoG_Handpose.mat",
        "stack": "MATLAB, TreeBagger (ported here to scikit-learn RandomForestClassifier)",
        "executed_by_us": "ported; her MATLAB was not run (toolboxes unavailable)",
        "protocol": {"window_s": list(MARIA_WINDOW), "bands_hz": [list(b) for b in MARIA_BANDS],
                     "features": "log band power per channel", "model": "random forest, 100 trees",
                     "cross_validation": "10-fold"},
        "published": PUBLISHED,
        "measured": measured,
        "patch_applies_to_pinned_commit": applies,
        "bugs": BUGS,
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2) + "\n")
    print(f"[{time.time() - t0:6.1f}s] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
