# Eidantz 2024, Riemannian tangent space

Upstream: <https://github.com/Eidantz/br41n.io_Hackathon_ECoG>, pinned at `e1b6f6a`
(11 October 2024), cloned by `./clone_prior_art.sh`. The default branch is `master` rather than `main`.

## The method

One file, `ecog_classification.py`, plus an electrode montage `pos.mat`. The recording is
fetched from Google Drive (id `1GRnZVuIRp7b3y3Ngvnm1DSLLFV_A1ZiI`), and that link no
longer serves the file, so our copy comes from the official g.tec archive instead.

Preprocessing is a 2 s trim, `filter(50., 300.)`, `notch_filter(np.arange(50, 351, 50))`
and epochs from 0 to 2 s. Classification is `Covariances` into `TSclassifier` with
`ShuffleSplit(10, test_size=0.3, random_state=30)`.

## No published numbers

The repository publishes none. We produced them, which is the only reason this project
appears in our comparison table at all.

## The problem with the task definition

The task is set up as one gesture against the other two:

```python
labels1[labels != j] = 0
```

With 30 trials per class, a constant answer of "not this gesture" is correct two thirds
of the time. 0.667 is the score before any learning. A one-vs-rest accuracy near 0.77
reads like a real 3-class accuracy near 0.77 while representing far less skill.

## What we measured

`run.py` uses the same epochs and the same preprocessing as everything else here, with
the majority-class baseline computed per fold and reported next to each accuracy:

| Gesture vs rest | Accuracy | Majority baseline | Gain |
|---|---|---|---|
| fist | 0.733 | 0.663 | +0.070 |
| peace sign | 0.719 | 0.678 | +0.041 |
| open hand | 0.867 | 0.659 | +0.208 |
| **mean** | **0.773** | **0.667** | **+0.106** |

The real gain over having no model is about 10 points.

## One fix the protocol needs

`Covariances(estimator="scm")` is singular on our epochs. Channels rejected by the
artefact criterion are zeroed before the common average reference, so afterwards they
are all equal to minus the average and the rows are identical. Those rows are dropped
before the covariance is estimated (channels 10, 20, 36). Without that the tangent-space
map fails outright, so the upstream protocol cannot run on these epochs as written.

## Running it

```bash
python prior-art/eidantz-helper/run.py
```

About 15 s, writing `metrics.json` next to the script. Needs pyriemann.
