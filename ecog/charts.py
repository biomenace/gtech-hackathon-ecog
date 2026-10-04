"""The six figures in the README, drawn from the result JSON.

Nothing here reads a number that was typed by hand: `figures.py` loads
`results/*.json` and the per-project `metrics.json` and hands rows to these
functions. If a result changes, re-run and the README follows.
"""

from __future__ import annotations

from .svg import (BAR, BASELINE, BLUE, GAP, GREY, GRID, INK, INK_2, MUTED, RAMP,
                  document, hbar, linear, ramp_color, readable_on, text)


def chart_accuracy_over_time(res) -> str:
    pts = res["accuracy_over_time"]
    W, H = 980, 470
    L, R, T, B = 64, 24, 40, 56
    xs = [p["t_center"] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = 0.0, 1.0

    def X(t):
        return L + (t - x0) / (x1 - x0) * (W - L - R)

    def Y(a):
        return T + (1 - (a - y0) / (y1 - y0)) * (H - T - B)

    out = []
    # the window the first pipeline used, and the one the data points at
    out.append(f'<rect x="{X(0.2):.1f}" y="{T}" width="{X(1.0)-X(0.2):.1f}" height="{H-T-B}" '
               f'fill="{BLUE}" opacity="0.10"/>')
    out.append(f'<rect x="{X(1.0):.1f}" y="{T}" width="{X(min(2.0,x1))-X(1.0):.1f}" height="{H-T-B}" '
               f'fill="{MUTED}" opacity="0.10"/>')
    for a in (0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(f'<line x1="{L}" y1="{Y(a):.1f}" x2="{W-R}" y2="{Y(a):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(text(L - 12, Y(a) + 5, f"{a:.1f}", 13, INK_2, "end", mono=True))
    # chance level
    out.append(f'<line x1="{L}" y1="{Y(1/3):.1f}" x2="{W-R}" y2="{Y(1/3):.1f}" stroke="{MUTED}" '
               f'stroke-width="1" stroke-dasharray="6 5"/>')
    out.append(text(W - R - 4, Y(1 / 3) - 9, "chance 0.33", 13, INK_2, "end"))
    # cue onset
    out.append(f'<line x1="{X(0):.1f}" y1="{T}" x2="{X(0):.1f}" y2="{H-B}" stroke="{BASELINE}" stroke-width="1"/>')
    out.append(text(X(0) + 6, T + 14, "cue", 13, INK_2))
    # axis
    out.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="{BASELINE}" stroke-width="1"/>')
    for t in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        if x0 <= t <= x1:
            out.append(text(X(t), H - B + 22, f"{t:.1f}", 13, INK_2, "middle", mono=True))
    out.append(text((L + W - R) / 2, H - 10, "time from cue onset, seconds", 14, INK_2, "middle"))
    # the line itself
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(p['t_center']):.1f},{Y(p['acc']):.1f}" for i, p in enumerate(pts))
    out.append(f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="2" stroke-linejoin="round"/>')
    # the peak, labelled
    peak = max(pts, key=lambda p: p["acc"])
    out.append(f'<circle cx="{X(peak["t_center"]):.1f}" cy="{Y(peak["acc"]):.1f}" r="6" fill="{BLUE}" '
               f'stroke="#ffffff" stroke-width="2"/>')
    out.append(text(X(peak["t_center"]) + 14, Y(peak["acc"]) - 6,
                    f"{peak['acc']:.3f} at {peak['t_center']:.2f} s", 15, INK, weight=600))
    out.append(text(X(0.2) + 8, H - B - 12, "the window we use: 0.2-1.0 s", 13, INK_2))
    out.append(text(X(1.0) + 8, T + 18, "the window version one used: 1.0-2.0 s", 13, INK_2))
    return document(W, H, "\n".join(out), "Decoding accuracy of a sliding 500 ms window against time from cue onset")


def chart_comparison(rows) -> str:
    W = 1140
    row_h = 54
    L, R, T = 410, 96, 24
    H = T + row_h * len(rows) + 46
    span = W - L - R
    X = linear(L, span)

    out = []
    for a in (0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{X(a):.1f}" y1="{T-6}" x2="{X(a):.1f}" y2="{T+row_h*len(rows):.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(text(X(a), T + row_h * len(rows) + 22, f"{a:.2f}", 13, INK_2, "middle", mono=True))
    out.append(f'<line x1="{X(1/3):.1f}" y1="{T-6}" x2="{X(1/3):.1f}" y2="{T+row_h*len(rows):.1f}" '
               f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="6 5"/>')
    out.append(text(X(1 / 3), T - 10, "chance", 13, INK_2, "middle"))
    out.append(f'<line x1="{L}" y1="{T-6}" x2="{L}" y2="{T+row_h*len(rows):.1f}" stroke="{BASELINE}" stroke-width="1"/>')
    for i, (label, sub, value, mine) in enumerate(rows):
        y = T + i * row_h + (row_h - BAR) / 2
        color = BLUE if mine else GREY
        out.append(hbar(L, y, X(value) - L, color))
        out.append(text(L - 14, y + 11, label, 15, INK if mine else INK_2, "end",
                        weight=600 if mine else 400))
        out.append(text(L - 14, y + 27, sub, 11.5, MUTED, "end"))
        out.append(text(X(value) + 10, y + 16, f"{value:.3f}", 15, INK, weight=600 if mine else 400, mono=True))
    return document(W, H, "\n".join(out), "Three-class gesture accuracy, our pipeline against prior work")


def chart_sweep(res2) -> str:
    sweep = res2["sweep_repeated1"]
    rows = ["0.2-0.7s", "0.3-0.8s", "0.5-1.0s", "0.2-1.0s", "0.3-1.3s", "0.5-1.5s", "1.0-2.0s",
            "multi3_early", "multi3_spread", "multi4"]
    cols = ["hg3", "hg6", "hg_wide", "broadband"]
    vals = [[sweep[f"{r}/{c}"] for c in cols] for r in rows]
    lo = min(min(v) for v in vals)
    hi = max(max(v) for v in vals)

    W = 760
    L, T, R, B = 158, 54, 16, 54
    cw = (W - L - R) / len(cols)
    ch = 40
    H = T + ch * len(rows) + B
    out = []
    for j, c in enumerate(cols):
        out.append(text(L + cw * (j + 0.5), T - 14, c, 13, INK_2, "middle"))
    for i, r in enumerate(rows):
        y = T + i * ch
        emph = r in ("0.5-1.0s", "multi3_spread")
        out.append(text(L - 12, y + ch / 2 + 5, r, 13, INK if emph else INK_2, "end", weight=600 if emph else 400))
        for j, c in enumerate(cols):
            v = vals[i][j]
            col = ramp_color(v, lo, hi)
            out.append(f'<rect x="{L+cw*j+GAP/2:.1f}" y="{y+GAP/2:.1f}" width="{cw-GAP:.1f}" '
                       f'height="{ch-GAP:.1f}" rx="3" fill="{col}"/>')
            out.append(text(L + cw * (j + 0.5), y + ch / 2 + 5, f"{v:.3f}", 13,
                            readable_on(col), "middle", mono=True))
    # ramp legend
    ly = H - 30
    lw = 170
    for k in range(len(RAMP)):
        out.append(f'<rect x="{L+k*lw/len(RAMP):.1f}" y="{ly}" width="{lw/len(RAMP)+0.6:.1f}" height="10" fill="{RAMP[k]}"/>')
    out.append(text(L - 12, ly + 9, "accuracy", 12, INK_2, "end"))
    out.append(text(L, ly + 26, f"{lo:.3f}", 11.5, MUTED, "start", mono=True))
    out.append(text(L + lw, ly + 26, f"{hi:.3f}", 11.5, MUTED, "end", mono=True))
    return document(W, H, "\n".join(out), "Ten time windows by four band sets, ten-fold accuracy in each cell")


def chart_confusion(cm, labels, title) -> str:
    n = len(labels)
    W = 560
    L, T, R, B = 132, 64, 24, 52
    cell = (W - L - R) / n
    H = T + cell * n + B
    total = sum(sum(r) for r in cm)
    out = [text(L, T - 36, "predicted", 13, INK_2)]
    for j, lab in enumerate(labels):
        out.append(text(L + cell * (j + 0.5), T - 14, lab, 14, INK_2, "middle"))
    out.append(text(L - 112, T + cell * n / 2, "true", 13, INK_2, "start"))
    hi = max(max(r) for r in cm)
    for i, lab in enumerate(labels):
        y = T + i * cell
        out.append(text(L - 14, y + cell / 2 + 5, lab, 14, INK, "end"))
        for j in range(n):
            v = cm[i][j]
            col = ramp_color(v, 0, hi)
            out.append(f'<rect x="{L+cell*j+GAP/2:.1f}" y="{y+GAP/2:.1f}" width="{cell-GAP:.1f}" '
                       f'height="{cell-GAP:.1f}" rx="4" fill="{col}"/>')
            out.append(text(L + cell * (j + 0.5), y + cell / 2 + 7, str(v), 20,
                            readable_on(col), "middle", weight=600, mono=True))
    out.append(text(L, H - 18, f"{total} test predictions, 50 folds", 13, MUTED))
    return document(W, H, "\n".join(out), title)


def chart_models(rows) -> str:
    W = 900
    row_h = 48
    L, R, T = 286, 84, 16
    H = T + row_h * len(rows) + 42
    span = W - L - R
    X = linear(L, span)

    out = []
    for a in (0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{X(a):.1f}" y1="{T-6}" x2="{X(a):.1f}" y2="{T+row_h*len(rows):.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(text(X(a), T + row_h * len(rows) + 22, f"{a:.2f}", 13, INK_2, "middle", mono=True))
    out.append(f'<line x1="{X(1/3):.1f}" y1="{T-6}" x2="{X(1/3):.1f}" y2="{T+row_h*len(rows):.1f}" '
               f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="6 5"/>')
    out.append(f'<line x1="{L}" y1="{T-6}" x2="{L}" y2="{T+row_h*len(rows):.1f}" stroke="{BASELINE}" stroke-width="1"/>')
    for i, (label, value, mine) in enumerate(rows):
        y = T + i * row_h + (row_h - BAR) / 2
        out.append(hbar(L, y, X(value) - L, BLUE if mine else GREY))
        out.append(text(L - 14, y + 16, label, 15, INK if mine else INK_2, "end", weight=600 if mine else 400))
        out.append(text(X(value) + 10, y + 16, f"{value:.3f}", 15, INK, weight=600 if mine else 400, mono=True))
    return document(W, H, "\n".join(out), "What we tried: accuracy of every model on the same folds")


def chart_glove(res) -> str:
    g = res["glove_regression_ridge"]["per_finger_r"]
    s = res["glove_regression_ridge"]["per_finger_r_std"]
    W = 760
    L, R, T = 110, 70, 20
    row_h = 52
    H = T + row_h * len(g) + 62
    span = W - L - R
    X = linear(L, span)

    out = []
    for a in (0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{X(a):.1f}" y1="{T-6}" x2="{X(a):.1f}" y2="{T+row_h*len(g):.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(text(X(a), T + row_h * len(g) + 22, f"{a:.2f}", 13, INK_2, "middle", mono=True))
    out.append(f'<line x1="{L}" y1="{T-6}" x2="{L}" y2="{T+row_h*len(g):.1f}" stroke="{BASELINE}" stroke-width="1"/>')
    for i, (finger, r) in enumerate(g.items()):
        cy = T + i * row_h + row_h / 2
        lo, hi = max(0.0, r - s[finger]), min(1.0, r + s[finger])
        whisker = (f'M{X(lo):.1f},{cy-7:.1f} V{cy+7:.1f} M{X(lo):.1f},{cy:.1f} H{X(hi):.1f} '
                   f'M{X(hi):.1f},{cy-7:.1f} V{cy+7:.1f}')
        out.append(f'<path d="{whisker}" stroke="{INK_2}" stroke-width="1.5" fill="none"/>')
        out.append(f'<circle cx="{X(r):.1f}" cy="{cy:.1f}" r="7" fill="{BLUE}" stroke="#ffffff" stroke-width="2"/>')
        out.append(text(L - 14, cy + 5, finger, 15, INK_2, "end"))
        out.append(text(X(hi) + 14, cy + 5, f"{r:.2f}", 15, INK, "start", mono=True))
    out.append(text(L, H - 8, "Pearson r, ten-fold; whiskers are one standard deviation across folds", 12.5, MUTED))
    return document(W, H, "\n".join(out), "Per-finger correlation between predicted and measured glove flexion")
