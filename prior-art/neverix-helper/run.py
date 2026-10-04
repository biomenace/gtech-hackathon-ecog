#!/usr/bin/env python3
"""Measure the neverix 2022 feature stack on a fixed protocol.

Their notebooks print 0.923-0.944, which is the maximum over several dozen
configurations read off the same resamples it was measured on. The same stack is
evaluated here once per protocol, with CSP fitted inside each fold: their own
16 x 3-fold scheme, and the 10-fold scheme every other number in this repository
uses.

Their topological features (giotto-tda: Takens embedding, Vietoris-Rips,
PersistenceEntropy) are not reproduced. That gap is stated in the README.

    python prior-art/neverix-helper/run.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ecog.prior import load_epochs, neverix_protocol

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


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
    measured = neverix_protocol(epochs, y)
    accs = [v["mean"] for v in measured.values()]

    Path(args.out).write_text(json.dumps({
        "upstream": "https://github.com/neverix/ecog-br41n-2022",
        "commit": "4833d1787dad0f1fe047bf5fe2bc7b4a1cad4925",
        "year": 2022,
        "dataset": "ECoG_Handpose.mat",
        "stack": "MNE, pyriemann, giotto-tda, scikit-learn",
        "executed_by_us": True,
        "window_s": [args.tmin, args.tmax],
        "bad_channels": bad.tolist(),
        "published": {
            "source": "cell outputs in MNE Analysis*.ipynb",
            "claimed_range": [0.923, 0.944],
            "caveat": "maximum over a configuration grid, selected on the same resamples it is measured on",
        },
        "measured": measured,
        "measured_range": [round(min(accs), 4), round(max(accs), 4)],
        "not_reproduced": ["giotto-tda topological features"],
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2) + "\n")
    print(f"[{time.time() - t0:6.1f}s] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
