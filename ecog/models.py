"""Classifiers, the cross-validation they are scored with, and the null test.

Everything is scored on `StratifiedKFold(10, shuffle=True, random_state=0)` unless
a function says otherwise, so two rows of any results table are comparable.

The model that wins is a shrinkage LDA. That is not an accident of tuning: 90
trials and a few hundred features is the regime where a regularised linear
discriminant beats everything with more capacity, and `search.py` measures the
alternatives rather than asserting this.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import FINGERS, epoch, log_band_power

OUTER = dict(n_splits=10, shuffle=True, random_state=0)
INNER = dict(n_splits=5, shuffle=True, random_state=1)


def lda():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


def logreg(c=0.1):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=4000, C=c))])


def classical_models():
    """The four models the baseline compares on identical folds."""
    return {
        "LDA-shrinkage": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "SVM-rbf": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
        "logreg": lambda: LogisticRegression(max_iter=4000, C=0.1),
        "random-forest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
    }


def cv_scores(make_clf, X, y, n_splits=10, seed=0, scale=True):
    """Returns (per-fold accuracies, summed confusion matrix, class order)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    classes = np.unique(y)
    accs, cm = [], np.zeros((len(classes), len(classes)), dtype=int)
    for tr, te in skf.split(X, y):
        steps = ([("scale", StandardScaler())] if scale else []) + [("clf", make_clf())]
        p = Pipeline(steps).fit(X[tr], y[tr]).predict(X[te])
        accs.append(accuracy_score(y[te], p))
        cm += confusion_matrix(y[te], p, labels=classes)
    return np.array(accs), cm, classes


def plain_cv(make_clf, X, y):
    """`cv_scores` without the confusion matrix, for tables of means."""
    accs = [accuracy_score(y[te], make_clf().fit(X[tr], y[tr]).predict(X[te]))
            for tr, te in StratifiedKFold(**OUTER).split(X, y)]
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs))}


def permutation_p(make_clf, X, y, observed, n_perm=200, n_splits=10, seed=0):
    """How often a shuffled label vector reaches `observed`. Floor is 1/(n_perm+1)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        acc, _, _ = cv_scores(make_clf, X, rng.permutation(y), n_splits=n_splits, seed=seed)
        hits += acc.mean() >= observed
    return (hits + 1) / (n_perm + 1)


def nested_choice(bank, y, make_clf=lda, outer=None, mapper=None):
    """Pick the feature set on inner folds, score it on the untouched outer fold.

    This is the only honest way to report a number that came out of a search, and
    it is the check no prior work on this recording did. `outer` overrides the fold
    scheme; `mapper` lets a caller spread the outer folds over processes, which
    changes the wall clock and not the result.
    """
    splits = list(outer if outer is not None else StratifiedKFold(**OUTER).split(y, y))
    keys = list(bank)

    def one_fold(split):
        tr, te = split
        scores = {}
        for k in keys:
            X = bank[k]
            inner = [accuracy_score(y[tr][ite], make_clf().fit(X[tr][itr], y[tr][itr]).predict(X[tr][ite]))
                     for itr, ite in StratifiedKFold(**INNER).split(X[tr], y[tr])]
            scores[k] = float(np.mean(inner))
        best = max(scores, key=scores.get)
        X = bank[best]
        return best, accuracy_score(y[te], make_clf().fit(X[tr], y[tr]).predict(X[te]))

    folds = list((mapper or map)(one_fold, splits))
    chosen = [f[0] for f in folds]
    accs = [f[1] for f in folds]
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "n_outer_folds": len(accs),
            "picked": {k: chosen.count(k) for k in sorted(set(chosen))}}


def riemannian(epochs, y, n_splits=10, seed=0):
    """Covariance-based classifiers, the family Eidantz 2024 used."""
    try:
        from pyriemann.classification import MDM
        from pyriemann.tangentspace import TangentSpace
        from pyriemann.estimation import Covariances
    except ImportError:
        return {"error": "pyriemann not installed"}

    cov = Covariances(estimator="oas").transform(epochs.astype(np.float64))
    out = {}
    for name, proto in (("MDM", MDM()),
                        ("TangentSpace+logreg", Pipeline([("ts", TangentSpace()),
                                                          ("lr", LogisticRegression(max_iter=4000))]))):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs = [accuracy_score(y[te], clone(proto).fit(cov[tr], y[tr]).predict(cov[te]))
                for tr, te in skf.split(cov, y)]
        out[name] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    return out


def csp_lda(epochs, y, n_components=4, n_splits=10, seed=0):
    """CSP is fitted inside each fold; fitting it once on all trials leaks."""
    try:
        from mne.decoding import CSP
    except ImportError:
        return {"error": "mne not installed"}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(epochs, y):
        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
        Xtr = csp.fit_transform(epochs[tr].astype(np.float64), y[tr])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(csp.transform(epochs[te].astype(np.float64)))))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs))}


def glove_regression(X, glove_epochs, n_splits=10, seed=0):
    """Ridge from ECoG features to mean per-finger flexion.

    R2 is reported alongside r because it is negative for three fingers: the model
    recovers the shape of the flexion and not its magnitude.
    """
    Y = glove_epochs.mean(axis=-1)
    corrs, r2s = [], []
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X):
        pred = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=10.0))]).fit(X[tr], Y[tr]).predict(X[te])
        corrs.append([0.0 if np.std(pred[:, i]) < 1e-12 or np.std(Y[te, i]) < 1e-12
                      else float(np.corrcoef(pred[:, i], Y[te, i])[0, 1])
                      for i in range(Y.shape[1])])
        ss_res = ((pred - Y[te]) ** 2).sum(axis=0)
        ss_tot = ((Y[te] - Y[te].mean(axis=0)) ** 2).sum(axis=0) + 1e-12
        r2s.append(1 - ss_res / ss_tot)
    corrs, r2s = np.array(corrs), np.array(r2s)
    return {
        "per_finger_r": {f: float(corrs[:, i].mean()) for i, f in enumerate(FINGERS)},
        "per_finger_r_std": {f: float(corrs[:, i].std()) for i, f in enumerate(FINGERS)},
        "per_finger_r2": {f: float(r2s[:, i].mean()) for i, f in enumerate(FINGERS)},
        "mean_r": float(corrs.mean()),
    }


def glove_leakage(glove_epochs, y, n_splits=10, seed=0):
    """The glove alone, as a classifier. Scores 1.0, which is why it stays a target."""
    accs, _, _ = cv_scores(lambda: LinearDiscriminantAnalysis(), glove_epochs.mean(axis=-1), y,
                           n_splits=n_splits, seed=seed)
    return {"mean": float(accs.mean()), "std": float(accs.std())}


def accuracy_over_time(clean, onsets, classes, bands, window=0.5, step=0.1, tmin=-1.0, tmax=3.0):
    """Slide one window across the trial and decode at each position.

    This is the measurement the whole result rests on: it is what showed that the
    window everyone used sits after the peak.
    """
    out, t = [], tmin
    while t + window <= tmax:
        ep, keep = epoch(clean, onsets, t, t + window)
        accs, _, _ = cv_scores(
            lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            log_band_power(ep, bands), classes[keep], n_splits=10)
        out.append({"t_center": round(t + window / 2, 3),
                    "acc": float(accs.mean()), "std": float(accs.std())})
        t += step
    return out
