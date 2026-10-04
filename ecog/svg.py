"""SVG primitives for the figures: palette, text, bars, the document wrapper.

One blue hue carries data (#2a78d6). Grey (#898781) is a de-emphasis slot, not a
category: it marks a bar that is not ours. It fails a chroma floor on purpose, and
every grey bar carries a direct numeric label, which is the relief that makes it
readable anyway. The 100-700 blue ramp is for the two heatmaps, where the colour
encodes magnitude.

Ink colours are written out literally rather than inherited, so the files render in
a plain SVG rasteriser and not only in a browser.
"""

from __future__ import annotations

BLUE = "#2a78d6"
GREY = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
MONO = 'ui-monospace, "Roboto Mono", monospace'

BAR = 22          # bar thickness, under the 24 px ceiling
RADIUS = 4        # rounded data end
GAP = 2           # surface gap between touching marks


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=15, fill=INK, anchor="start", weight=400, mono=False, extra=""):
    family = MONO if mono else FONT
    nums = ' style="font-variant-numeric:tabular-nums"' if mono else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{family}\' font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{nums}{extra}>{esc(s)}</text>')


def hbar(x0, y, width, color, thickness=BAR):
    """Square at the baseline, 4 px rounded at the data end."""
    r = min(RADIUS, max(width, 0))
    if width <= r:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(width,0):.1f}" height="{thickness}" fill="{color}"/>'
    return (f'<path d="M{x0:.1f},{y:.1f} H{x0+width-r:.1f} a{r},{r} 0 0 1 {r},{r} '
            f'V{y+thickness-r:.1f} a{r},{r} 0 0 1 -{r},{r} H{x0:.1f} Z" fill="{color}"/>')


def linear(origin, span):
    """Map a value in 0..1 to a pixel position along `span`."""
    def place(v):
        return origin + v * span

    return place


def ramp_color(v, lo, hi):
    if hi <= lo:
        return RAMP[-1]
    t = (v - lo) / (hi - lo)
    return RAMP[max(0, min(len(RAMP) - 1, int(round(t * (len(RAMP) - 1)))))]


def readable_on(color):
    """Ink for a label sitting on a ramp cell."""
    return "#ffffff" if (RAMP.index(color) if color in RAMP else 0) >= 6 else "#0b0b0b"


def document(w, h, body, title):
    """Explicit pixel size as well as the viewBox: GitHub renders an image reference
    through an <img>, which needs an intrinsic size."""
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{esc(title)}" xmlns="http://www.w3.org/2000/svg">\n{body}\n</svg>\n')
