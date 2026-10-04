#!/usr/bin/env python3
"""Audit the bkowshik project: how much of it was ever run, and what we take from it.

There is no accuracy to reproduce, because the project executed no benchmark. That is
a claim about someone else's work, so it is counted rather than asserted: every
notebook is walked for code cells, cells marked `#| eval: false`, and cells carrying
an output.

Two things here are ours to use, and both are verified rather than trusted. The clone holds
the recording in its git-lfs store, which is our fallback if the official g.tec archive is
unavailable, so the checksum of whatever was fetched is recorded.
The literature ranges we compare against come from `nbs/09_benchmark.ipynb`, so every
bound we quote must appear verbatim in that notebook or this script fails.

    python prior-art/bkowshik-helper/run.py

Reads only, fits nothing, under a second.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTEBOOKS = HERE / "br41n-ecog-hand-pose" / "nbs"
EVAL_FALSE = re.compile(r"#\|\s*eval:\s*false")

# Quoted in the README. Each bound must be literal text in 09_benchmark.ipynb.
LITERATURE = {
    "ecog_high_gamma_lda": (0.80, 0.95),
    "eegnet_4class_eeg": (0.65, 0.75),
    "shallowconvnet_ecog": (0.80, 0.92),
    "riemannian": (0.80, 0.90),
    "finger_regression_linear_r": (0.45, 0.60),
    "finger_regression_deep_cnn_r": (0.55, 0.70),
}
CITATIONS = ["Liang", "Bougrain", "Xie"]


def audit_notebooks(nbdir):
    """Count code cells, suppressed cells and cells with an output, per notebook."""
    totals = {"code_cells": 0, "eval_false": 0, "cells_with_output": 0}
    per_file = {}
    for path in sorted(nbdir.glob("*.ipynb")):
        cells = [c for c in json.loads(path.read_text())["cells"] if c.get("cell_type") == "code"]
        counts = {
            "code_cells": len(cells),
            "eval_false": sum(bool(EVAL_FALSE.search("".join(c.get("source", [])))) for c in cells),
            "with_output": sum(bool(c.get("outputs")) for c in cells),
        }
        per_file[path.name] = counts
        totals["code_cells"] += counts["code_cells"]
        totals["eval_false"] += counts["eval_false"]
        totals["cells_with_output"] += counts["with_output"]
    return {**totals, "per_notebook": per_file}


def missing_from_benchmark(notebook):
    """Bounds and citations we quote that are not in their benchmark notebook."""
    text = "\n".join("".join(c.get("source", [])) for c in json.loads(notebook.read_text())["cells"])
    missing = [f"{name}:{bound}" for name, bounds in LITERATURE.items() for bound in bounds
               if f"{bound:.2f}" not in text and f"{bound:g}" not in text]
    return missing + [c for c in CITATIONS if c not in text]


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def record(cells, dataset):
    return {
        "upstream": "https://github.com/bkowshik/br41n-ecog-hand-pose",
        "commit": "41ad6d2de0cac7038b3a1ee6f6c899e79ff10cce",
        "year": 2026,
        "dataset": "ECoG_Handpose.mat",
        "stack": "nbdev, MNE, scikit-learn, PyTorch (EEGNet), Ridge",
        "executed_by_us": False,
        "why_not_executed": "there is nothing to execute: no code cell carries an output, and the "
                            "benchmark cells are marked '#| eval: false'",
        "published": {"source": "none", "claimed": None,
                      "note": "the repository holds no accuracy, correlation or confusion matrix"},
        "notebook_audit": cells,
        "value_to_us": {
            "dataset_source": {
                "path_in_clone": "data/ECoG_Handpose.mat", "transport": "git-lfs",
                "fetch": "../../fetch_dataset.sh --local",
                "note": "the fallback source. The official g.tec archive is the primary one, and "
                        "the Google Drive link in the Eidantz repository no longer serves the file",
                **dataset,
            },
            "literature_ranges": {
                "source": "nbs/09_benchmark.ipynb", "verified_verbatim": True,
                **{k: list(v) for k, v in LITERATURE.items()},
                "citations": ["Liang & Bougrain 2012", "Xie 2018"],
            },
        },
        "unused_modules": {
            "regression.py": "Ridge on the glove channels, written but never executed, so "
                             "continuous finger flexion on this recording had not been attempted "
                             "before us",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(HERE / "metrics.json"))
    args = ap.parse_args()

    t0 = time.time()
    if not NOTEBOOKS.is_dir():
        print("the bkowshik project is not cloned yet; run ./clone_prior_art.sh first",
              file=sys.stderr)
        return 1

    cells = audit_notebooks(NOTEBOOKS)
    print(f"notebooks {len(cells['per_notebook'])}, code cells {cells['code_cells']}, "
          f"eval:false {cells['eval_false']}, with output {cells['cells_with_output']}", flush=True)

    missing = missing_from_benchmark(NOTEBOOKS / "09_benchmark.ipynb")
    if missing:
        print(f"quoted literature is NOT in 09_benchmark.ipynb: {missing}", file=sys.stderr)
        return 1
    print(f"literature: {len(LITERATURE)} ranges and {len(CITATIONS)} citations found verbatim",
          flush=True)

    data = Path(args.data)
    dataset = checksum(data) if data.is_file() else {}
    print(f"dataset: {dataset or 'not fetched yet, run fetch_dataset.sh'}", flush=True)

    out = record(cells, dataset)
    out["runtime_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{time.time() - t0:6.1f}s] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
