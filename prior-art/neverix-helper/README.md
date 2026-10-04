# neverix 2022, a feature union and a grid search

Upstream: <https://github.com/neverix/ecog-br41n-2022>, pinned at `4833d17`
(9 May 2022), cloned by `./clone_prior_art.sh`. BR41N.IO Spring School 2022.

The strongest number anyone has claimed on this recording, and the one we had to
address directly: the notebooks print up to 0.944.

## The method

The notebooks are `MNE Analysis.ipynb`, `MNE Analysis-Copy1.ipynb`,
`MNE Analysis High Gamma.ipynb` and `EDA.ipynb`. Preprocessing is `raw.filter(100, 500)`,
average reference, 50 Hz notch. Features are a `FeatureUnion` of hand-written
transformers (`BandPower`, `PSDWelch`, `ERPFeature`, `TopoFormer`) plus
`XdawnCovariances` from pyriemann and topological features from giotto-tda, built on a
Takens embedding and Vietoris-Rips complexes (`Amplitude`, `PersistenceEntropy`,
`NumberOfPoints`).
Evaluation is 16 resamples of a 3-fold split, swept over CSP component counts from 4 to
28 and dozens of feature combinations.

## The claim and what is wrong with it

Printed in the notebook cells: best 0.923, 0.944, 0.936 and 0.926. Welch features reach
0.828 and 0.836. CSP alone reaches 0.50 to 0.55, ERP 0.37 to 0.46, and a dummy
classifier 0.265.

The 0.923 to 0.944 range is the maximum over the configuration grid, read off the same
resamples it was measured on. It is a selection statistic rather than an expected
accuracy, and nothing in the notebooks holds out a split for the choice of
configuration.

This is not an accusation of error. It is the standard failure mode of a 48-hour
hackathon, and the upstream work is careful in every other respect. It does mean the
number cannot be compared with a single-protocol number, including ours.

## What we measured

`run.py` takes the same feature stack, evaluated once per protocol with no grid maximum:
log band power in four overlapping bands (40-100, 80-140, 120-180, 160-220 Hz), binned
Welch PSD at 10 Hz and 5 Hz resolution, and CSP at 4, 8 and 12 components fitted inside
each fold.

| Protocol | Model | Accuracy |
|---|---|---|
| 16 x 3-fold (theirs) | PCA40+LDA | 0.758 ± 0.067 |
| 16 x 3-fold (theirs) | PCA40+LDA+CSP | 0.767 ± 0.065 |
| 16 x 3-fold (theirs) | logreg C=0.1 | 0.763 ± 0.063 |
| 16 x 3-fold (theirs) | logreg C=0.1+CSP | 0.770 ± 0.064 |
| 10-fold (ours) | PCA40+LDA | 0.767 ± 0.105 |
| 10-fold (ours) | PCA40+LDA+CSP | 0.756 ± 0.120 |
| 10-fold (ours) | logreg C=0.1 | 0.756 ± 0.130 |
| 10-fold (ours) | logreg C=0.1+CSP | 0.778 ± 0.111 |

0.756 to 0.778, on their own resampling scheme as well as ours. The gap to 0.923-0.944
is the cost of the grid maximum.

The PCA in front of the LDA is about cost rather than accuracy. The feature union is
about 3600-dimensional for 60 training trials, and a Ledoit-Wolf LDA at that width costs
roughly 115 s per fit, which makes the 48-fold protocol infeasible. Sixty training
trials span at most 59 dimensions, so `PCA(40)` discards nothing that could have been
estimated.

## The gap we state ourselves

We did not reproduce the giotto-tda topological features. Our reproduction covers band
power, Welch PSD and CSP. If the topological features carry the difference, this
comparison understates the upstream result, though the notebooks' own per-feature
numbers do not suggest that: the best single families they report are band power and
Welch.

## Running it

```bash
python prior-art/neverix-helper/run.py
```

About 17 min on one CPU core, writing `metrics.json` next to the script.
