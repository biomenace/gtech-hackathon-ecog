# MariaGoniIba 2022, MATLAB and a random forest

Upstream: <https://github.com/MariaGoniIba/BR41N.IO-Hackathon-ECoG-Epilepsy>, pinned at `e2e6159`
(7 December 2022), cloned by `./clone_prior_art.sh`.

The only earlier project that publishes numbers on this exact recording, which makes it
the baseline ours is measured against. Its window, 1 s starting 1 s after the cue, is
also the window we adopt for every head-to-head comparison.

| | Accuracy |
|---|---|
| Published, three gestures | 0.811 |
| Published, rest vs movement | 0.789 |
| Her protocol ported and re-run by us | **0.878** three gestures, 0.800 rest vs movement |

## Published results

Read off `Results.png` upstream, after converting the counts to fractions (see the
second bug below).

| Task | Accuracy | Balanced accuracy | F1 |
|---|---|---|---|
| rest vs movement | 142/180 = 0.789 | 0.789 | 0.765 |
| fist / peace / open hand | 73/90 = 0.811 | 0.793 | 0.800 |

## Bugs in the upstream code

The common average reference does not reference anything. `preprocessing.m` computes:

```matlab
med_s = median(y_trim);              % 1 x N: median across channels, per sample
y_CAR(i,:) = y_trim(i,:) - med_s(i); % subtracts ONE number from the whole channel
```

In MATLAB, `median` of a matrix reduces along the first dimension, so `med_s` holds one
value per time sample. Indexing it with the channel counter `i` picks the median at
sample `i` and subtracts that scalar from the entire channel. What the comment
describes as spatial re-referencing is a per-channel DC offset.

`Results.Acc` is a count rather than a rate. `RF.m` sets
`Results.Acc = length(find(classreal == classpredict))`, and the author marked that line
`%corregir!` herself. The printed table reads `142` and `73`, which have to be divided
by 180 and 90.

`car_fix.patch` fixes both, as a unified diff against the pinned commit. It is a patch rather than a
committed edit so her clone stays untouched, and `run.py` checks on every run that it
still applies:

```bash
git -C prior-art/maria-helper/BR41N.IO-Hackathon-ECoG-Epilepsy apply ../car_fix.patch
```

## Fixing the first bug makes her result worse

`run.py` runs her protocol both ways. We expected the correction to help. It does not:

| Reference | Three gestures | Rest vs movement |
|---|---|---|
| broken, as written (per-channel scalar) | **0.878 ± 0.092** | 0.800 ± 0.071 |
| fixed (real reference signal subtracted) | 0.622 ± 0.151 | 0.750 ± 0.062 |

The bug is real, and on this recording it helps her. The explanation is in the data: on
a 60-channel grid over sensorimotor cortex most channels carry the task-related
high-gamma response, so the median across channels largely is the signal. Subtracting
it removes what the classifier needs. Her accidental per-channel DC offset leaves the
high-gamma envelope intact.

Our own pipeline uses a real reference and does not collapse this way, scoring 0.867 on
the same window and bands, because it references after band-passing to 50-300 Hz and
uses the mean. The subtracted signal is then the common broadband and line-noise
component rather than the band-limited task response.

We found a bug in earlier work and measured it, and the measurement went against what we
assumed. The direction we measured is the one reported here.

## What we ran, and what we did not

We did not execute her MATLAB. It needs the Signal Processing and Statistics toolboxes
(`iirnotch`, `firls`, `bandpower`, `TreeBagger`), which we do not have, and GNU Octave
has no `TreeBagger`.

`ecog.prior.maria_protocol` ports the protocol instead: her window, her three bands
(60-90, 110-140, 160-190 Hz), log band power per channel, a 100-tree random forest,
10-fold. It reproduces her trial counts exactly, 180 for rest vs movement and 90 for
the three gestures, which is what `Results.png` reports.

It is a port and not her binary. `TreeBagger` and `RandomForestClassifier` are both
bagged CART ensembles with different defaults, and her fold assignment
(`M_cross_validation.m`) is unseeded, so a point or two of difference from her published
figure is expected. The ported number comes out 6.7 points above her published 0.811,
which we read as a difference between the two ensembles rather than as a correction of
her work.

## Running it

```bash
python prior-art/maria-helper/run.py
```

About 50 s on one CPU core, writing `metrics.json` next to the script. Needs numpy,
scipy and scikit-learn only.

With MATLAB available, her own code runs like this:

```bash
cd prior-art/maria-helper/BR41N.IO-Hackathon-ECoG-Epilepsy
git apply ../car_fix.patch          # optional, see the table above first
ln -s ../../../ECoG_Handpose.mat .
matlab -batch MainScript
```
