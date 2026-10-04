#!/usr/bin/env python3
"""Check every number the README claims against the files the runs wrote.

The README is prose, so its numbers are typed by hand. This is what keeps them
honest: each claim below names the key it has to come from, the value is read out of
`results/*.json` or `prior-art/*/metrics.json`, and the two are compared at the
precision the README quotes. A run that changes a result and a README that was not
updated with it will not both pass.

    python verify.py            # 0 if the README matches the results, 1 if not
    python verify.py --verbose  # print every claim, not only the failures

It reads files and fits nothing. To regenerate the results, run the seven commands
the README lists; to regenerate the figures, `python -m ecog.figures`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOL = 5e-4  # the README quotes three decimals


def load(rel):
    return json.loads((ROOT / rel).read_text())


def read_all():
    return {
        "base": load("results/baseline.json"),
        "search": load("results/search.json"),
        "final": load("results/final.json"),
        "maria": load("prior-art/maria-helper/metrics.json"),
        "neverix": load("prior-art/neverix-helper/metrics.json"),
        "eidantz": load("prior-art/eidantz-helper/metrics.json"),
        "bkowshik": load("prior-art/bkowshik-helper/metrics.json"),
        "official": load("prior-art/official-baseline.json"),
        "recording": load("results/recording.json"),
    }


def our_claims(base, search, final):
    fixed = final["fixed/multi3_spread/hg6"]
    nested = final["nested_full_bank"]
    lda3 = base["gesture_3class"]["LDA-shrinkage"]
    acm = [v["mean"] for v in search["acm"]["plain"].values()]
    return [
        ("headline fixed pipeline", fixed["mean"], 0.971),
        ("headline fixed pipeline, std", fixed["std"], 0.062),
        ("headline fixed pipeline, folds", fixed["n_folds"], 50),
        ("nested over the full bank", nested["mean"], 0.963),
        ("nested, std", nested["std"], 0.060),
        ("nested, outer folds", nested["n_outer_folds"], 30),
        ("nested picks six bands",
         sum(n for k, n in nested["picked"].items() if k.endswith("hg6")), 29),
        ("first version, LDA", lda3["mean"], 0.867),
        ("first version, std", lda3["std"], 0.083),
        ("first version, random forest", base["gesture_3class"]["random-forest"]["mean"], 0.856),
        ("first version, logreg", base["gesture_3class"]["logreg"]["mean"], 0.811),
        ("first version, RBF SVM", base["gesture_3class"]["SVM-rbf"]["mean"], 0.744),
        ("permutation p, first version", base["permutation_test"]["p"], 0.00498),
        ("permutation p, fixed pipeline", final["permutation"]["p_value"], 0.00498),
        ("permutation, observed", final["permutation"]["observed_single_10fold"], 0.978),
        ("accuracy peak, centre", base["accuracy_peak"]["t_center"], 0.75),
        ("accuracy peak", base["accuracy_peak"]["acc"], 0.956),
        ("accuracy peak, std", base["accuracy_peak"]["std"], 0.074),
        ("best single feature set", search["sweep_best"]["mean"], 0.978),
        ("best single set is 0.5-1.0s/hg6", search["sweep_best"]["key"], "0.5-1.0s/hg6"),
        ("bank size", len(final["sweep_repeated1"]), 40),
        ("rest vs movement, first version", base["rest_vs_movement"]["mean"], 0.744),
        ("rest vs movement, final", final["rest_vs_movement"]["rest-1.2_-0.2"]["mean"], 0.799),
        ("rest vs movement, final std", final["rest_vs_movement"]["rest-1.2_-0.2"]["std"], 0.084),
        ("rest vs movement, early rest", final["rest_vs_movement"]["rest-2.0_-1.0"]["mean"], 0.784),
        ("glove leakage check", base["glove_baseline_lda"]["mean"], 1.0),
        ("glove regression, mean r", base["glove_regression_ridge"]["mean_r"], 0.605),
        ("ensemble", search["ensemble"]["mean"], 0.922),
        ("ensemble best member", max(search["ensemble"]["per_view"].values()), 0.944),
        ("histogram gradient boosting", search["trees"]["hist_gradient_boosting"]["mean"], 0.822),
        ("random forest, search round", search["trees"]["random_forest"]["mean"], 0.889),
        ("augmented covariance, worst", min(acm), 0.789),
        ("augmented covariance, best", max(acm), 0.833),
        ("augmented covariance, nested", search["acm"]["nested"]["mean"], 0.822),
        ("sub-window voting", search["subwindow_vote"]["mean"], 0.944),
        ("EEGNet", base["eegnet"]["mean"], 0.556),
        ("trials", base["n_trials"], 90),
        ("rejected channels", base["bad_channels"], [10, 20, 36]),
    ]


def prior_art_claims(maria, neverix, eidantz, bkowshik):
    runs = list(neverix["measured"].values())
    lit = bkowshik["value_to_us"]["literature_ranges"]
    audit = bkowshik["notebook_audit"]
    return [
        ("MariaGoniIba, published", maria["published"]["three_gestures"]["accuracy"], 0.811),
        ("MariaGoniIba, published rest", maria["published"]["rest_vs_movement"]["accuracy"], 0.789),
        ("MariaGoniIba re-run, broken reference",
         maria["measured"]["broken"]["three_gestures"]["acc"], 0.878),
        ("MariaGoniIba re-run, fixed reference",
         maria["measured"]["fixed"]["three_gestures"]["acc"], 0.622),
        ("MariaGoniIba re-run, rest vs movement",
         maria["measured"]["broken"]["rest_vs_movement"]["acc"], 0.800),
        ("MariaGoniIba trial counts",
         [maria["measured"]["broken"]["rest_vs_movement"]["n_trials"],
          maria["measured"]["broken"]["three_gestures"]["n_trials"]], [180, 90]),
        ("car_fix.patch applies to the pinned commit",
         maria["patch_applies_to_pinned_commit"], True),
        ("neverix, claimed", neverix["published"]["claimed_range"], [0.923, 0.944]),
        ("neverix, re-run low", min(v["mean"] for v in runs), 0.756),
        ("neverix, re-run high", max(v["mean"] for v in runs), 0.778),
        ("Eidantz, fist vs rest", eidantz["measured"]["gesture_1_vs_rest"]["acc"], 0.733),
        ("Eidantz, peace vs rest", eidantz["measured"]["gesture_2_vs_rest"]["acc"], 0.719),
        ("Eidantz, open hand vs rest", eidantz["measured"]["gesture_3_vs_rest"]["acc"], 0.867),
        ("Eidantz, mean", eidantz["measured"]["overall_one_vs_rest"], 0.773),
        ("Eidantz, gain over the no-skill floor", eidantz["gain_over_no_skill"],
         round(eidantz["measured"]["overall_one_vs_rest"] - 2 / 3, 4)),
        ("bkowshik published nothing", bkowshik["published"]["claimed"], None),
        ("bkowshik code cells", audit["code_cells"], 84),
        ("bkowshik cells with an output", audit["cells_with_output"], 0),
        ("bkowshik cells marked eval: false", audit["eval_false"], 26),
        ("bkowshik notebooks", len(audit["per_notebook"]), 11),
        ("literature ranges checked against the notebook", lit["verified_verbatim"], True),
        ("literature, high gamma + LDA", lit["ecog_high_gamma_lda"], [0.80, 0.95]),
        ("literature, linear finger regression", lit["finger_regression_linear_r"], [0.45, 0.60]),
        ("dataset byte count",
         bkowshik["value_to_us"]["dataset_source"]["bytes"], 123246843),
        ("dataset checksum", bkowshik["value_to_us"]["dataset_source"]["sha256"],
         "b2b65f0f040ee9663c523801d92ba7e685d2ade94a2c1592c9975320966fab60"),
    ]


def recording_claims(rec):
    """CH62 carries the cue, not a device's classification. Measured, not assumed."""
    cue = rec["cue_channel"]
    return [
        ("the only discrete channel is CH62", rec["discrete_channels"], [62]),
        ("CH62 distinct values", cue["distinct_values"], [0, 1, 2, 3]),
        ("cue blocks", cue["blocks"], 90),
        ("cue blocks per class", cue["blocks_per_class"], {"1": 30, "2": 30, "3": 30}),
        ("cue block length, seconds", cue["block_seconds"], [2.0, 2.04]),
        ("gap length, seconds", cue["gap_seconds"], [2.08, 12.08]),
        ("class-to-class transitions", cue["class_to_class_transitions"], 0),
        ("recording shape", rec["shape"], [67, 507025]),
        ("recording duration, seconds", rec["duration_s"], 422.5),
        ("recording checksum", rec["sha256"],
         "b2b65f0f040ee9663c523801d92ba7e685d2ade94a2c1592c9975320966fab60"),
    ]


def official_claims(official):
    """The baseline from the paper that ships with the dataset, quoted from its Table 4."""
    s1 = official["accuracies_on_S1_rps"]
    return [
        ("official baseline, best method", s1["best"]["acc"], 0.824),
        ("official baseline, best method std", s1["best"]["std"], 0.020),
        ("official, whitened LDA + CSP", s1["whitened"]["LDA+CSP"]["acc"], 0.738),
        ("official, whitened LDA + channel selection",
         s1["whitened"]["LDA+channel_selection"]["acc"], 0.691),
        ("official, whitened rLDA + PCA", s1["whitened"]["rLDA+PCA"]["acc"], 0.654),
        ("official, rLDA vectorised", s1["no_whitening"]["rLDA_vectorised"]["acc"], 0.688),
        ("official, LDA + CSP", s1["no_whitening"]["LDA+CSP"]["acc"], 0.638),
        ("official, LDA + channel selection",
         s1["no_whitening"]["LDA+channel_selection"]["acc"], 0.522),
        ("official, subject identified", official["subject_identification"]["subject"], "S1"),
        ("official, trials per class", official["dataset"]["trials_per_class"], 30),
        ("official paper doi", official["paper"]["doi"], "10.3389/fnins.2019.00901"),
        ("official archive checksum", official["dataset"]["sha256"],
         "b2b65f0f040ee9663c523801d92ba7e685d2ade94a2c1592c9975320966fab60"),
    ]


def delta_claims(base, final, maria, neverix, official):
    """The README states every delta against the nested estimate, not the fixed one."""
    nested = final["nested_full_bank"]["mean"]
    runs = [v["mean"] for v in neverix["measured"].values()]
    return [
        ("delta over the best published number",
         nested - maria["published"]["three_gestures"]["accuracy"], 0.152),
        ("delta over the best earlier method re-run",
         nested - maria["measured"]["broken"]["three_gestures"]["acc"], 0.085),
        ("delta over our own first version",
         nested - base["gesture_3class"]["LDA-shrinkage"]["mean"], 0.096),
        ("delta over neverix re-run, low", nested - max(runs), 0.185),
        ("delta over neverix re-run, high", nested - min(runs), 0.207),
        ("delta over the official baseline",
         nested - official["accuracies_on_S1_rps"]["best"]["acc"], 0.139),
    ]


def matrix_claims(base, final):
    """Confusion matrices quoted verbatim in the README."""
    return [
        ("final pipeline confusion", final["fixed/multi3_spread/hg6"]["confusion_sum"],
         [[144, 6, 0], [2, 148, 0], [0, 5, 145]]),
        ("first version confusion", base["gesture_3class"]["LDA-shrinkage"]["confusion"],
         [[24, 3, 3], [3, 27, 0], [3, 0, 27]]),
        ("rest vs movement confusion",
         final["rest_vs_movement"]["rest-1.2_-0.2"]["confusion_sum"],
         [[400, 50], [131, 319]]),
    ]


def range_claims(base):
    """cuDNN convolution kernels are not deterministic, so the EEGNet fold spread moves
    between GPUs even though the mean does not. It is checked as a range."""
    return [("EEGNet, std across folds", base["eegnet"]["std"], 0.13, 0.16)]


def matches(got, want):
    if isinstance(want, bool) or isinstance(got, bool):
        return got == want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(float(got) - float(want)) <= TOL
    return got == want


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    r = read_all()

    exact = (our_claims(r["base"], r["search"], r["final"])
             + prior_art_claims(r["maria"], r["neverix"], r["eidantz"], r["bkowshik"])
             + recording_claims(r["recording"])
             + official_claims(r["official"])
             + delta_claims(r["base"], r["final"], r["maria"], r["neverix"], r["official"])
             + matrix_claims(r["base"], r["final"]))
    ranges = range_claims(r["base"])

    stale = []
    for name, got, want in exact:
        ok = matches(got, want)
        if not ok:
            stale.append((name, got, want))
        if args.verbose:
            print(f"{'ok  ' if ok else 'FAIL'}  {name}: README {want}, results {got}")
    for name, got, lo, hi in ranges:
        ok = lo <= got <= hi
        if not ok:
            stale.append((name, round(got, 4), f"{lo} to {hi}"))
        if args.verbose:
            print(f"{'ok  ' if ok else 'FAIL'}  {name}: README {lo} to {hi}, results {got}")

    total = len(exact) + len(ranges)
    print(f"\n{total - len(stale)}/{total} README claims match the results in this repository.")
    for name, got, want in stale:
        print(f"  STALE: {name}, README says {want}, results say {got}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
