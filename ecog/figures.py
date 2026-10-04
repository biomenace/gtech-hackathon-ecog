"""Draw every figure in the README from the result files.

    python -m ecog.figures          -> results/*.svg

Our numbers come from `results/*.json`; prior-art numbers come from the
`metrics.json` of the helper that measured each project, so a figure cites the run
that produced the number rather than a copy of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .charts import (chart_accuracy_over_time, chart_comparison, chart_confusion,
                     chart_glove, chart_models, chart_sweep)

ROOT = Path(__file__).resolve().parents[1]
HELPERS = {"maria": "maria-helper", "neverix": "neverix-helper", "eidantz": "eidantz-helper"}
OFFICIAL = "official-baseline.json"


def comparison_rows(base, final, prior, official):
    """Three-class accuracy, ours and everyone else's, on the same recording."""
    neverix = max(v["mean"] for v in prior["neverix"]["measured"].values())
    return [
        ("Ours, fixed pipeline", "multi-window high gamma + shrinkage LDA, 50 folds",
         final["fixed/multi3_spread/hg6"]["mean"], True),
        ("Ours, nested CV", "feature choice inside the loop, 30 outer folds",
         final["nested_full_bank"]["mean"], True),
        ("Our first version", "single 1.0-2.0 s window, three bands",
         base["gesture_3class"]["LDA-shrinkage"]["mean"], False),
        ("MariaGoniIba 2022, re-run", "her protocol ported to Python by us",
         prior["maria"]["measured"]["broken"]["three_gestures"]["acc"], False),
        ("MariaGoniIba 2022, published", "her own figure, MATLAB, not re-run", 
         prior["maria"]["published"]["three_gestures"]["accuracy"], False),
        ("Official baseline, TVLDA", "Gruenwald et al. 2019, their S1 row",
         official["accuracies_on_S1_rps"]["best"]["acc"], False),
        ("neverix 2022", "re-run by us on the same folds", neverix, False),
        ("Eidantz 2024", "re-run by us, one-vs-rest, no-skill is 0.667",
         prior["eidantz"]["measured"]["overall_one_vs_rest"], False),
    ]


def model_rows(base, search, final):
    """Everything we tried, including what lost."""
    return [
        ("Shrinkage LDA, hg6 (ours)", final["fixed/multi3_spread/hg6"]["mean"], True),
        ("Soft-voting ensemble of five", search["ensemble"]["mean"], False),
        ("Random forest", search["trees"]["random_forest"]["mean"], False),
        ("Riemannian tangent space", base["riemannian"]["TangentSpace+logreg"]["mean"], False),
        ("Augmented covariance, best", max(v["mean"] for v in search["acm"]["plain"].values()), False),
        ("Histogram gradient boosting", search["trees"]["hist_gradient_boosting"]["mean"], False),
        ("EEGNet, trained from scratch", base["eegnet"]["mean"], False),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--prior-art", default=str(ROOT / "prior-art"))
    args = ap.parse_args()
    results, prior_dir = Path(args.results), Path(args.prior_art)

    def read(path):
        return json.loads(Path(path).read_text())

    base = read(results / "baseline.json")
    search = read(results / "search.json")
    final = read(results / "final.json")
    prior = {k: read(prior_dir / d / "metrics.json") for k, d in HELPERS.items()}
    official = read(prior_dir / OFFICIAL)

    figures = {
        "accuracy-over-time": chart_accuracy_over_time(base),
        "sweep": chart_sweep(final),
        "confusion": chart_confusion(final["fixed/multi3_spread/hg6"]["confusion_sum"],
                                     ["fist", "peace", "open"],
                                     "Confusion matrix of the final pipeline over fifty folds"),
        "glove": chart_glove(base),
        "comparison": chart_comparison(comparison_rows(base, final, prior, official)),
        "models": chart_models(model_rows(base, search, final)),
    }
    for name, svg in figures.items():
        (results / f"{name}.svg").write_text(svg)
        print(f"wrote {results / f'{name}.svg'} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
