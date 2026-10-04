# bkowshik 2026, the most complete scaffold and no results

Upstream: <https://github.com/bkowshik/br41n-ecog-hand-pose>, pinned at `41ad6d2`
(4 May 2026), cloned by `./clone_prior_art.sh`.

## The method

The best-organised project on this dataset: nbdev, eleven notebooks, a real module
layout with `preprocessing.py`, `epoching.py`, `features.py` (mu, beta, low gamma, high
gamma), `classification.py`, `dl.py` for EEGNet and `regression.py` for Ridge on the
glove channels.

## Nothing was ever run

`run.py` walks every notebook in the clone and counts:

| | |
|---|---|
| notebooks | 11 |
| code cells | 84 |
| cells marked `#\| eval: false` | 26 |
| **cells carrying an output** | **0** |

Every accuracy, correlation and confusion matrix the project would have produced is
absent. That is a claim about someone else's work, so it is measured rather than
asserted, and re-running the script gives the same counts. This project appears in our comparison table
without a number, which is the honest entry rather than an omission on our side.

It is still the most useful of the four to us, for two reasons.

## A second source for the recording

The 123 MB file is committed through git-lfs at `data/ECoG_Handpose.mat`, byte for byte
the same file as the official g.tec archive. The root `fetch_dataset.sh` prefers the official
download and falls back to here, which is useful because the Google
Drive link in the Eidantz repository no longer serves the file:

```bash
./clone_prior_art.sh
./fetch_dataset.sh --local
```

Expected: 123 246 843 bytes, sha256
`b2b65f0f040ee9663c523801d92ba7e685d2ade94a2c1592c9975320966fab60`. A mismatch stops the
script with both values printed. Every number in this repository was computed on exactly
that file.

## The literature ranges we compare against

`nbs/09_benchmark.ipynb` assembles published ranges for this kind of decoding. We use
them as the external yardstick, since one patient in one session is not enough to claim
anything about generality.

`run.py` checks the quotation. Every bound below has to appear verbatim in that notebook,
along with both citations, or the script exits non-zero. The table is a verified
quotation rather than a trusted transcription.

| Approach | Reported range |
|---|---|
| ECoG high gamma + LDA | 0.80 to 0.95 |
| EEGNet, 4-class EEG | 0.65 to 0.75 |
| ShallowConvNet on ECoG | 0.80 to 0.92 |
| Riemannian methods | 0.80 to 0.90 |
| Finger decoding, linear (Liang & Bougrain, 2012) | r = 0.45 to 0.60 |
| Finger decoding, deep CNN (Xie, 2018) | r = 0.55 to 0.70 |

Our 0.963 nested and 0.971 fixed sit at or above the top of the high-gamma range, and
our finger regression r = 0.605 sits in the upper part of the linear range. Our EEGNet
result of 0.556 falls below the published EEGNet range, which we read as a sample-size
effect: 90 epochs is not enough to train it.

`regression.py` exists upstream but was never run, so continuous finger flexion on this
recording had not been attempted before us.

## Running the audit

```bash
python prior-art/bkowshik-helper/run.py
```

Under a second. Reads only and fits nothing. Writes `metrics.json` next to the script, holding the
cell counts, the checksum of the fetched recording, and the literature ranges with their
verification flag.
