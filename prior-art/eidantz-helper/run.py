#!/usr/bin/env python3
"""Measure the Eidantz 2024 protocol on this recording.

The upstream repository publishes no numbers, so this runner produces them. It also
prints the majority baseline next to every accuracy, because the task is one gesture
against the other two and that scores 0.667 before any learning.

    python prior-art/eidantz-helper/run.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ecog.prior import eidantz_protocol, load_epochs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SINGULAR_COVARIANCE = (
    "Covariances(estimator='scm') is singular here: channels rejected before the common "
    "average reference become identical afterwards. Those rows are dropped before "
    "covariance estimation."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(HERE / "metrics.json"))
    ap.add_argument("--tmin", type=float, default=1.0)
    ap.add_argument("--tmax", type=float, default=2.0)
    args = ap.parse_args()

    t0 = time.time()
    epochs, y, bad = load_epochs(args.data, args.tmin, args.tmax)
    print(f"epochs {epochs.shape}, bad channels {bad.tolist()}", flush=True)
    measured = eidantz_protocol(epochs, y, bad)

    Path(args.out).write_text(json.dumps({
        "upstream": "https://github.com/Eidantz/br41n.io_Hackathon_ECoG",
        "commit": "e1b6f6aa7aa0f1a1c0635f55a9adf1f1ee1adccb",
        "default_branch": "master",
        "year": 2024,
        "dataset": "ECoG_Handpose.mat",
        "stack": "MNE, pyriemann Covariances + TSclassifier, scikit-learn",
        "executed_by_us": True,
        "window_s": [args.tmin, args.tmax],
        "bad_channels": bad.tolist(),
        "published": {"source": "none", "claimed": None},
        "measured": measured,
        "gain_over_no_skill": round(measured["overall_one_vs_rest"] - 2 / 3, 4),
        "fix_required": SINGULAR_COVARIANCE,
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2) + "\n")
    print(f"[{time.time() - t0:6.1f}s] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
