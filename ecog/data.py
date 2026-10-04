"""Reading the recording, cleaning it, cutting it into trials, and turning trials
into features.

One module because there is one path: every number in this repository, ours and
every prior-art protocol we re-ran, goes through these functions. A comparison
between two methods therefore differs by the method and by nothing else.

The recording is `ECoG_Handpose.mat`, 67 channels at 1200 Hz. In the numbering its own
documentation uses: CH1 is sample time, CH2 to CH61 are the ECoG grid over right
sensorimotor cortex, CH62 is the paradigm label (0 relax, 1 fist, 2 peace, 3 open hand),
CH63 to CH67 are data-glove flexion from thumb to little finger. Zero-based row indices
below are one lower.

The glove is a regression target only. An LDA trained on it alone scores 1.0, because the
glove is the instrument that measures the gestures.
"""

from __future__ import annotations

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch

FS = 1200
GESTURES = {1: "fist", 2: "peace", 3: "open"}
FINGERS = ["thumb", "index", "middle", "ring", "little"]

HIGH_GAMMA_3 = [(60, 90), (110, 140), (160, 190)]
HIGH_GAMMA_6 = [(60, 90), (90, 120), (120, 150), (150, 180), (180, 210), (210, 240)]


def load(path):
    """Returns (ecog, cue_labels, glove) as float32: CH2-61, CH62, CH63-67."""
    y = loadmat(path)["y"].astype(np.float32)
    return y[1:61], y[61], y[62:67]


def bandpass(x, low, high, fs=FS, order=4):
    b, a = butter(order, [low, high], btype="band", fs=fs)
    return filtfilt(b, a, x, axis=-1).astype(np.float32)


def notch_cascade(x, fs=FS, line=50.0, harmonics=6, q=30.0):
    """Mains and its harmonics, up to Nyquist."""
    out = x
    for k in range(1, harmonics + 1):
        f = line * k
        if f >= fs / 2:
            break
        b, a = iirnotch(f, q, fs=fs)
        out = filtfilt(b, a, out, axis=-1).astype(np.float32)
    return out


def bad_channels(x, threshold=5.0):
    """Channels whose standard deviation is a MAD outlier. Returns 10, 20, 36 here."""
    s = x.std(axis=1)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-12
    return np.where(np.abs(s - med) / mad > threshold)[0]


def preprocess(ecog, band=(50.0, 300.0)):
    """Reject, notch, band-pass, then reference. Returns (clean, bad_channels).

    The reference subtracts the mean across the grid, which is a reference signal.
    The published MATLAB pipeline subtracts one scalar per channel instead, and
    `prior-art/maria-helper/` measures what that difference is worth.
    """
    bad = bad_channels(ecog)
    x = ecog.copy()
    if len(bad):
        x[bad] = 0.0
    x = notch_cascade(x)
    x = bandpass(x, *band)
    return x - x.mean(axis=0, keepdims=True), bad


def cue_onsets(labels):
    """Rising edges out of rest. Returns (sample indices, gesture per onset)."""
    labels = np.asarray(labels)
    onsets = np.where((labels[1:] != 0) & (labels[:-1] == 0))[0] + 1
    return onsets, labels[onsets].astype(int)


def cue_structure(labels, fs=FS):
    """Describe CH62, so "it is the stimulus marker" is measured rather than assumed.

    Some hackathon packs ship a device's own classification next to a hand-made ground
    truth, and ask you to beat the device. This recording does not: CH62 is the only
    discrete channel in the file, and it is a rectangle held for exactly as long as the
    cue photograph was on screen. A classifier output would not be.
    """
    lab = np.rint(np.asarray(labels)).astype(int)
    bounds = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
    runs = [(int(lab[a]), int(b - a)) for a, b in zip(bounds[:-1], bounds[1:])]
    cues = [n for c, n in runs if c != 0]
    gaps = [n for c, n in runs if c == 0]
    classes, counts = np.unique([c for c, _ in runs if c != 0], return_counts=True)
    return {
        "distinct_values": sorted(set(lab.tolist())),
        "blocks": len(cues),
        "blocks_per_class": {str(k): int(v) for k, v in zip(classes, counts)},
        "block_seconds": [round(min(cues) / fs, 2), round(max(cues) / fs, 2)],
        "gap_seconds": [round(min(gaps) / fs, 2), round(max(gaps) / fs, 2)],
        "class_to_class_transitions": sum(1 for i in range(1, len(runs))
                                          if runs[i - 1][0] != 0 and runs[i][0] != 0),
    }


def epoch(x, onsets, tmin, tmax, fs=FS):
    """Cut [tmin, tmax) around each onset. Returns (epochs, which onsets fitted)."""
    a, b = int(round(tmin * fs)), int(round(tmax * fs))
    n = b - a
    keep = (onsets + a >= 0) & (onsets + b <= x.shape[-1])
    return np.stack([x[..., s + a : s + a + n] for s in onsets[keep]]), keep


def log_band_power(epochs, bands, fs=FS):
    """Log power per channel per band: the feature every headline number uses."""
    feats = []
    for low, high in bands:
        f = bandpass(epochs.reshape(-1, epochs.shape[-1]), low, high, fs=fs)
        feats.append(np.log(np.mean(f.reshape(epochs.shape) ** 2, axis=-1) + 1e-12))
    return np.concatenate(feats, axis=-1)


def welch_features(epochs, low, high, nfft=256, fs=FS):
    """Log Welch PSD, every bin kept. Wide: 3360 features for 90 trials."""
    freqs, psd = welch(epochs, fs=fs, nperseg=min(nfft, epochs.shape[-1]), axis=-1)
    mask = (freqs >= low) & (freqs <= high)
    return np.log(psd[..., mask] + 1e-20).reshape(epochs.shape[0], -1)


CHANNEL_MAP = {
    "CH1": "sample time",
    "CH2-CH61": "ECoG, raw and DC-coupled, right sensorimotor cortex",
    "CH62": "paradigm label: 0 relax, 1 fist, 2 peace, 3 open hand",
    "CH63-CH67": "data glove, thumb to little finger",
}


def main():
    """Describe the recording itself, with no model involved.

        python -m ecog.data        -> results/recording.json

    Cheap: it loads the file, counts channel cardinalities and measures the cue train.
    """
    import argparse
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(root / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(root / "results" / "recording.json"))
    args = ap.parse_args()

    path = Path(args.data)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    raw = loadmat(args.data)["y"]
    cardinality = [int(len(np.unique(row))) for row in raw]
    discrete = [i + 1 for i, n in enumerate(cardinality) if n <= 8]

    out = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "shape": list(raw.shape),
        "sampling_rate_hz": FS,
        "duration_s": round(float(raw.shape[1]) / FS, 1),
        "channel_map": CHANNEL_MAP,
        "discrete_channels": discrete,
        "channel_cardinality": cardinality,
        "cue_channel": cue_structure(raw[61]),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"{path.name}: {out['shape']} at {FS} Hz, {out['duration_s']} s")
    print(f"discrete channels: {discrete}  (everything else is continuous)")
    print(f"cue train: {out['cue_channel']['blocks']} blocks, "
          f"{out['cue_channel']['blocks_per_class']}, "
          f"{out['cue_channel']['block_seconds']} s each, "
          f"{out['cue_channel']['class_to_class_transitions']} class-to-class transitions")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
