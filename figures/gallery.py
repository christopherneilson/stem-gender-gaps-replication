#!/usr/bin/env python3
r"""Port of the Claude Design figure system to Python, so the paper's figures
rebuild from the committed panel data instead of being hand-exported.

The design was authored as an HTML/JS prototype (`Paper Figures.dc.html` in the
handoff bundle) that renders each figure as SVG with millimetre geometry and
rasterises it at export. This module reproduces that SVG generation. Geometry,
palette and type sizes are transcribed from the prototype; the numbers come from
`figures/data/*.csv`, the same panel files the manuscript already uses.

Two deliberate departures from the prototype, both bugs there:

1. Country order. The prototype defines the shared ORDER but several figures
   override it with a sort on their own values (`sortBy` on the dumbbell, the
   funnel's own final share). That gives the paper three different orderings and
   contradicts the response letter's answer to editor comment 8. Everything here
   uses ORDER.

2. The subscripts in the decomposition axis titles. Unicode has no subscript
   "f", so the prototype used U+2091 (subscript e) as a stand-in and the axis
   read ln(T_m/T_e). SVG has real subscripts, so they are set with <tspan>.

Outputs SVG and vector PDF via svglib, with Ghostscript embedding the fonts.
PDFs are replaced only after a successful export; failures exit nonzero.

Run:  python figures/gallery.py
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "07_revision" / "02_data" / "harmonized" / "figures"

# ---------------------------------------------------------------- constants
PT = 0.352778                      # 1 typographic point in mm
# The prototype names a CSS fallback stack, "'Helvetica Neue', Helvetica, Arial,
# sans-serif". Do not restore it: svglib tries to resolve every family in the
# list and hangs indefinitely on the larger appendix figures. A single name
# parses in 0.08s, and Helvetica is one of the PDF base-14 fonts, so it is the
# face the stack resolves to in print anyway.
SANS = "Helvetica"
SZ = {"tick": 8 * PT, "axis": 9 * PT, "letter": 10 * PT, "note": 8 * PT,
      "small": 7 * PT}

NEUTRALS = {"ink": "#1A1815", "tick": "#4A463F", "soft": "#7D786F",
            "rule": "#8C877D", "faint": "#CFCBC2"}
PALETTES = {
    # The main text draws on two accents only, teal and coral, so the appendix
    # uses the same pair: "female" was a third colour (sand #D2915F) that
    # appeared nowhere in the paper and made the appendix read as a separate
    # document. "third" survives because figB8 and figD1 carry three series and
    # the main text has no three-series figure to borrow a colour from.
    "Teal and coral": {"male": "#2A6F6B", "female": "#C0614A", "neg": "#C0614A",
                       "third": "#4E6E8C", "gap": "#454340"},
    "Okabe-Ito": {"male": "#0072B2", "female": "#E69F00", "neg": "#D55E00",
                  "third": "#009E73", "gap": "#5B5750"},
}
C = dict(NEUTRALS, **PALETTES["Teal and coral"])

W_DOUBLE, H_BAR, ROW_H = 190.0, 74.0, 5.6

# The one order every exhibit uses (editor comment 8).
ORDER = ["Taiwan", "Chile", "Brazil", "China", "Uganda", "Finland", "Spain",
         "Australia", "Greece", "Sweden"]


# ------------------------------------------------------------- svg helpers
def r3(n: float) -> str:
    return f"{round(float(n), 3):g}"


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, size=None, fill=None, anchor=None, weight=None, rotate=None,
         raw=False) -> str:
    body = str(s) if raw else esc(s)
    if raw and "<tspan" in body:
        return _rich_text(x, y, body, size or SZ["tick"], fill or C["ink"],
                          anchor, rotate)
    a = [f'x="{r3(x)}"', f'y="{r3(y)}"',
         f'font-size="{r3(size or SZ["tick"])}"',
         f'font-family="{SANS}"', f'fill="{fill or C["ink"]}"']
    if anchor:
        a.append(f'text-anchor="{anchor}"')
    if weight:
        a.append(f'font-weight="{weight}"')
    if rotate:
        a.append(f'transform="rotate({rotate} {r3(x)} {r3(y)})"')
    return f'<text {" ".join(a)}>{body}</text>'


def _rich_text(x, y, markup, size, fill, anchor, rotate) -> str:
    """Expand tspan-subscript markup into individually positioned <text> runs.

    Both PDF converters we can reach (svglib and cairosvg) mis-position tspans
    inside an anchored <text>: each run is re-anchored around the parent x, so
    the label comes apart -- Figure 3's axis titles rendered as "ln(T   ) (log
    points)" with the subscripts scattered into the gap. Plain positioned
    <text> elements survive every renderer, so we measure each run with the
    exact Helvetica base-14 metrics (reportlab is already a dependency via
    svglib; widths scale linearly with size, so mm in, mm out) and lay the runs
    end to end ourselves. Subscripts drop 0.22 em, matching what
    baseline-shift="sub" was meant to do.
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth
    runs, pos = [], 0
    for m in re.finditer(r'<tspan[^>]*font-size="([0-9.]+)"[^>]*>(.*?)</tspan>',
                         markup):
        if m.start() > pos:
            runs.append((markup[pos:m.start()], size, False))
        runs.append((m.group(2), float(m.group(1)), True))
        pos = m.end()
    if pos < len(markup):
        runs.append((markup[pos:], size, False))

    def unesc(t):
        return (t.replace("&lt;", "<").replace("&gt;", ">")
                .replace("&amp;", "&"))

    widths = [stringWidth(unesc(t), "Helvetica", sz) for t, sz, _ in runs]
    total = sum(widths)
    cur = {"middle": x - total / 2, "end": x - total}.get(anchor, x)
    parts = []
    for (t, sz, sub), w in zip(runs, widths):
        dy = 0.22 * size if sub else 0.0
        parts.append(f'<text x="{r3(cur)}" y="{r3(y + dy)}" '
                     f'font-size="{r3(sz)}" font-family="{SANS}" '
                     f'fill="{fill}">{t}</text>')
        cur += w
    g = "".join(parts)
    if rotate:
        return f'<g transform="rotate({rotate} {r3(x)} {r3(y)})">{g}</g>'
    return g


def sub_title(head: str, num: str, den: str, tail: str) -> str:
    """Axis title with real subscripts: head + ln(num/den) + tail.

    The prototype wrote these with Unicode subscript letters, which do not exist
    for "f"; it fell back to subscript e and the label came out as ln(T_m/T_e).
    The tspan markup produced here is expanded into positioned runs by
    _rich_text() at render time; it never reaches the SVG verbatim.
    """
    def v(sym, sb):
        return (f'{esc(sym)}<tspan baseline-shift="sub" '
                f'font-size="{r3(SZ["axis"] * 0.72)}">{esc(sb)}</tspan>')
    return (f'{esc(head)}ln({v(num[0], num[1])}/{v(den[0], den[1])}){esc(tail)}')


def line(x1, y1, x2, y2, stroke=None, pt=0.6, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{r3(x1)}" y1="{r3(y1)}" x2="{r3(x2)}" y2="{r3(y2)}" '
            f'stroke="{stroke or C["ink"]}" stroke-width="{r3(pt * PT)}"{d} />')


def rect(x, y, w, h, fill, extra="") -> str:
    if h <= 0 or w <= 0:
        return ""
    return (f'<rect x="{r3(x)}" y="{r3(y)}" width="{r3(w)}" height="{r3(h)}" '
            f'fill="{fill}"{extra} />')


def circle(x, y, r, fill, stroke=None, sw=0.35) -> str:
    st = f' stroke="{stroke}" stroke-width="{r3(sw)}"' if stroke else ""
    return f'<circle cx="{r3(x)}" cy="{r3(y)}" r="{r3(r)}" fill="{fill}"{st} />'


def poly(pts, fill, stroke=None, sw=0.4) -> str:
    p = " ".join(f"{r3(a)},{r3(b)}" for a, b in pts)
    st = f' stroke="{stroke}" stroke-width="{r3(sw)}"' if stroke else ""
    return f'<polygon points="{p}" fill="{fill}"{st} />'


def fmt_val(v, kind) -> str:
    if kind == "pct":
        return f"{round(v)}%"
    if kind == "signed":
        sign = "+" if v > 0 else ("-" if v < 0 else "")
        return f"{sign}{abs(round(v))}"
    if kind == "signed1":
        sign = "+" if v > 0 else ("-" if v < 0 else "")
        return f"{sign}{abs(v):.1f}"
    if kind == "dec1":
        return f"{v:.1f}"
    return str(round(v))


def nice_ticks(lo, hi):
    raw = (hi - lo) / 4
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = None
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            step = m * mag
            break
    step = step or 10 * mag
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(0.0 if abs(v) < 1e-9 else v)
        v += step
    return out


def pad_range(vals):
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.12 or 1
    return [lo - pad, hi + pad]


def wrap(s, max_chars):
    lines, cur = [], ""
    for w in str(s).split():
        if cur and len(cur + " " + w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def series_fill(name):
    n = str(name).lower()
    if n.startswith("female minus male"):
        return C["gap"]
    if n.startswith("female"):
        return C["female"]
    if n.startswith("male"):
        return C["male"]
    if n.startswith("pipeline"):
        return C["male"]
    if n.startswith("stem first"):
        return C["female"]
    if n.startswith("any-stem"):
        return C["third"]
    # Threshold series (figB8). Matched exactly, not by prefix: "top 10%"
    # startswith "top 1%" is false but "top 1" is a prefix of both, and a
    # startswith test in the wrong order would silently colour them alike.
    if n in ("top 1%", "top 5%", "top 10%"):
        return {"top 1%": C["male"], "top 5%": C["third"],
                "top 10%": C["female"]}[n]
    return C["gap"]


def legend_swatch(x, y, fill, label, size):
    return (rect(x, y - size * 0.72, size * 0.9, size * 0.8, fill)
            + text(x + size * 1.25, y, label, size=size))


def axes(g, ticks, y_title, fmt, show_y=True):
    s = line(g["x0"], g["y0"], g["x0"], g["y1"], stroke=C["rule"], pt=0.55)
    s += line(g["x0"], g["y0"], g["x1"], g["y0"], stroke=C["rule"], pt=0.55)
    for t in ticks:
        y = g["sy"](t)
        s += line(g["x0"] - 0.8, y, g["x0"], y, stroke=C["rule"], pt=0.55)
        if show_y:
            s += text(g["x0"] - 1.4, y + SZ["tick"] * 0.36, fmt_val(t, fmt),
                      anchor="end", size=SZ["tick"], fill=C["tick"])
    if y_title:
        plot_h = g["y0"] - g["y1"]
        est = len(re.sub("<[^>]+>", "", y_title)) * SZ["axis"] * 0.52
        mid = len(y_title) // 2
        cut = y_title.rfind(" ", 0, mid)
        if cut <= 0:
            cut = y_title.find(" ", mid)
        if est > plot_h and cut > 0 and "<" not in y_title:
            for i, l in enumerate([y_title[:cut], y_title[cut + 1:]]):
                s += text(3.2 + i * SZ["axis"] * 1.15,
                          (g["y0"] + g["y1"]) / 2, l, anchor="middle",
                          size=SZ["axis"], rotate=-90)
        else:
            s += text(g["yTitleX"], (g["y0"] + g["y1"]) / 2, y_title,
                      anchor="middle", size=SZ["axis"], rotate=-90,
                      raw="<" in y_title)
    return s


# ------------------------------------------------------------------ panels
def dumbbell_panel(p, rows, W, H):
    # Margins are tight on both sides: the longest country label ends ~4mm from
    # the left edge, and the change column sits right of the axis with the same
    # ~4mm clear on the right, so the canvas centres on its ink. Figure 1 passes
    # a reduced W so the plot area itself keeps its width.
    lab_x, x0, x1, mT = 15.0, 19.0, W - 22.0, 13.0
    d_col = W - 4.0
    x_min, x_max = p["x"]["min"], p["x"]["max"]

    def sx(v):
        return x0 + (v - x_min) / (x_max - x_min) * (x1 - x0)

    a, b = p["pairs"][0][0], p["pairs"][1][0]
    # ORDER, not a sort on the values (see module docstring).
    present = set(rows["x"])
    cats = [c for c in ORDER if c in present]

    def val(c, ser):
        m = rows[(rows.x == c) & (rows.s == ser)]
        return None if m.empty else float(m.v.iloc[0])

    def dot(x, y):
        return circle(x, y, 1.25, C["male"])

    def dia(x, y):
        d = 1.5
        return poly([(x, y - d), (x + d, y), (x, y + d), (x - d, y)],
                    "#FFFFFF", C["neg"], 0.5)

    s = ""
    y_bot = mT + ROW_H * len(cats) + 3.2
    for rl in p.get("ref", []):
        s += line(sx(rl["at"]), mT - 3, sx(rl["at"]), y_bot, pt=0.7)
        s += text(sx(rl["at"]) + 1.4, y_bot - 1.2, rl["label"],
                  size=SZ["small"], fill=C["soft"])

    kx = x0
    for glyph, label in ((dia, p["pairs"][1][1]), (dot, p["pairs"][0][1])):
        s += glyph(kx + 1.5, mT - 6.4)
        s += text(kx + 4.2, mT - 6.4 + SZ["small"] * 0.36, label,
                  size=SZ["small"], fill=C["tick"])
        kx += 4.2 + len(label) * SZ["small"] * 0.56 + 7
    s += text(d_col, mT - 6.4 + SZ["small"] * 0.36, "change (pp)",
              anchor="end", size=SZ["small"], fill=C["soft"])

    for i, c in enumerate(cats):
        y = mT + ROW_H * (i + 0.5)
        va, vb = val(c, a), val(c, b)
        s += text(lab_x, y + SZ["tick"] * 0.36, c, anchor="end", size=SZ["tick"])
        if va is None or vb is None:
            continue
        lo, hi = min(va, vb), max(va, vb)
        s += line(sx(lo), y, sx(hi), y, pt=1.1, stroke=C["faint"])
        s += dia(sx(lo), y) if vb == lo else dot(sx(lo), y)
        s += dia(sx(hi), y) if vb == hi else dot(sx(hi), y)
        s += text(sx(lo) - 2.6, y + SZ["tick"] * 0.36, f"{lo:.1f}",
                  anchor="end", size=SZ["small"], fill=C["tick"])
        s += text(sx(hi) + 2.6, y + SZ["tick"] * 0.36, f"{hi:.1f}",
                  size=SZ["small"], fill=C["tick"])
        d = va - vb
        s += text(d_col, y + SZ["tick"] * 0.36, fmt_val(d, "signed1"),
                  anchor="end", size=SZ["tick"],
                  fill=C["male"] if d >= 0 else C["neg"])

    t = x_min
    while t <= x_max + 1e-9:
        s += line(sx(t), y_bot, sx(t), y_bot + 0.8, stroke=C["rule"], pt=0.55)
        s += text(sx(t), y_bot + SZ["tick"] + 1.5, round(t), anchor="middle",
                  size=SZ["tick"], fill=C["tick"])
        t += 10
    s += line(x0, y_bot, x1, y_bot, stroke=C["rule"], pt=0.55)
    s += text((x0 + x1) / 2, y_bot + SZ["tick"] + SZ["axis"] + 3.4,
              p["xTitle"], anchor="middle", size=SZ["axis"])
    return s


def hbar_panel(p, rows, W, H, levels):
    lab_x = p["gutter"] - 2 if p.get("paired") else 30.0
    x0 = p["gutter"] if p.get("paired") else 50.0
    x1 = p["gutter"] + p["span"] if p.get("paired") else W - 12.0
    col_x = p["colX"] if p.get("paired") else 46.0
    mT = 6.0 if p.get("paired") else 8.0
    x_min, x_max = (p["xr"] if p.get("paired") else (p["y"]["min"], p["y"]["max"]))

    def sx(v):
        return x0 + (v - x_min) / (x_max - x_min) * (x1 - x0)

    present = set(rows["x"])
    data = [(c, float(rows[rows.x == c].v.iloc[0]))
            for c in ORDER if c in present]

    def lv(c):
        if levels is None:
            return None
        m = levels[(levels.x == c) & (levels.s == "female")]
        return None if m.empty else float(m.v.iloc[0])

    s = ""
    for i, (name, v) in enumerate(data):
        y = mT + ROW_H * (i + 0.5)
        zero, xv = sx(0), sx(v)
        if not p.get("noNames"):
            s += text(lab_x, y + SZ["tick"] * 0.36, name, anchor="end",
                      size=SZ["tick"])
        l = lv(name)
        if l is not None:
            s += text(col_x, y + SZ["tick"] * 0.36, round(l), anchor="end",
                      size=SZ["tick"], fill=C["soft"])
        neg = v < 0
        s += rect(min(zero, xv), y - ROW_H * 0.3, abs(xv - zero), ROW_H * 0.6,
                  C["neg"] if neg else C["male"])
        # A long positive bar pushes its value label into the level column
        # (Sweden's +32 collided with its grey 62). When the label would cross
        # the column, set it inside the bar end in white instead.
        from reportlab.pdfbase.pdfmetrics import stringWidth
        val = fmt_val(v, "signed")
        wlab = stringWidth(val, "Helvetica", SZ["tick"])
        inside = (not neg) and lv(name) is not None \
            and xv + 1.2 + wlab > col_x - 5.0
        if inside:
            s += text(xv - 1.2, y + SZ["tick"] * 0.36, val, anchor="end",
                      size=SZ["tick"], fill="#FFFFFF")
        else:
            s += text(min(zero, xv) - 1.2 if neg else xv + 1.2,
                      y + SZ["tick"] * 0.36, val,
                      anchor="end" if neg else "start", size=SZ["tick"])

    y_bot = mT + ROW_H * len(data)
    t = math.ceil(x_min / 20) * 20
    while t <= x_max + 1e-9:
        s += line(sx(t), y_bot, sx(t), y_bot + 0.8, stroke=C["rule"], pt=0.55)
        if not p.get("noX"):
            s += text(sx(t), y_bot + SZ["tick"] + 1.5, fmt_val(t, "signed"),
                      anchor="middle", size=SZ["tick"], fill=C["tick"])
        t += 20
    s += line(x0, y_bot, x1, y_bot, stroke=C["rule"], pt=0.55)
    s += line(sx(0), mT - 1.5, sx(0), y_bot, pt=0.7)
    if p.get("paired"):
        s += text(p["vTitleX"], (mT + y_bot) / 2, p["colTitle"],
                  anchor="middle", size=SZ["small"], fill=C["soft"], rotate=-90)
    if p.get("xTitle") and not p.get("noX"):
        s += text((x0 + x1) / 2, y_bot + SZ["tick"] + SZ["axis"] + 3.4,
                  p["xTitle"], anchor="middle", size=SZ["axis"])
    if p.get("letter"):
        s += text(0.5, 3.2, p["letter"], size=SZ["letter"], weight="700")
    return s


def decomp_panel(p, rows, W, H):
    mL, mR, mT, mB = 16.0, 6.0, 6.0, 15.0
    g = {"x0": mL, "x1": W - mR, "y0": H - mB, "y1": mT, "yTitleX": 6.6}
    plot_w, plot_h = g["x1"] - g["x0"], g["y0"] - g["y1"]
    xs, ys = list(rows.xn), list(rows.v)
    x_min, x_max = pad_range(xs + [0])
    y_min, y_max = pad_range(ys + [0])

    def sx(v):
        return g["x0"] + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v):
        return g["y0"] - (v - y_min) / (y_max - y_min) * plot_h

    g["sy"] = sy
    s = (f'<clipPath id="clipd"><rect x="{r3(g["x0"])}" y="{r3(g["y1"])}" '
         f'width="{r3(plot_w)}" height="{r3(plot_h)}" /></clipPath>')

    mk = [(sx(a), sy(b)) for a, b in zip(xs, ys)]
    cg, clabels = "", []
    for share in (0.55, 0.60, 0.65, 0.70, 0.75):
        tot = math.log(share / (1 - share))
        cg += line(sx(x_min), sy(tot - x_min), sx(x_max), sy(tot - x_max),
                   pt=0.5, stroke="#E0E0E0")
        xs0, xs1 = max(x_min, tot - y_max), min(x_max, tot - y_min)
        if xs1 - xs0 < (x_max - x_min) * 0.12:
            continue
        best, best_d = None, -1.0
        f = 0.08
        while f <= 0.92:
            xv = xs0 + f * (xs1 - xs0)
            px, py = sx(xv), sy(tot - xv)
            f += 0.03
            if px < g["x0"] + 9 or px > g["x1"] - 9 or py < g["y1"] + 3 or py > g["y0"] - 3:
                continue
            d = min([math.hypot(m[0] - px, m[1] - py) for m in mk] or [1e9])
            for c in clabels:
                d = min(d, math.hypot(c[0] - px, c[1] - py) * 0.6)
            if d > best_d:
                best_d, best = d, (px, py)
        if best and best_d > 4.2:
            clabels.append((best[0], best[1], f"{round(share * 100)}% male"))
    s += f'<g clip-path="url(#clipd)">{cg}</g>'
    s += line(g["x0"], sy(0), g["x1"], sy(0), pt=0.5, dash="1.2 0.9")
    s += line(sx(0), g["y1"], sx(0), g["y0"], pt=0.5, dash="1.2 0.9")
    s += axes(g, nice_ticks(y_min, y_max), p["yTitle"], "dec1")
    for t in nice_ticks(x_min, x_max):
        s += line(sx(t), g["y0"], sx(t), g["y0"] + 0.8, stroke=C["rule"], pt=0.55)
        s += text(sx(t), g["y0"] + SZ["tick"] + 1.4, f"{t:.1f}",
                  anchor="middle", size=SZ["tick"], fill=C["tick"])
    for cx, cy, t in clabels:
        s += text(cx, cy, t, size=SZ["small"], fill=C["soft"], anchor="middle")

    pts = sorted(((sx(r.xn), sy(r.v), r.s, r.xn) for r in rows.itertuples()),
                 key=lambda q: q[0])
    for cx, cy, name, xn in pts:
        s += text(cx + 2.0, cy + SZ["small"] * 0.36, name, size=SZ["small"])
        s += circle(cx, cy, 1.15, C["neg"] if xn >= 0 else C["male"],
                    "#FFFFFF", 0.3)
    s += text(g["x0"] + 1, g["y0"] - 1.6, "pipeline favors women",
              size=SZ["small"], fill=C["soft"])
    s += text(g["x1"] - 1, g["y0"] - 1.6, "pipeline favors men",
              size=SZ["small"], fill=C["soft"], anchor="end")
    s += text((g["x0"] + g["x1"]) / 2, H - 1.2, p["xTitle"], anchor="middle",
              size=SZ["axis"], raw="<" in p["xTitle"])
    return s


def marker(kind, x, y, fill, stroke):
    d = 1.25
    if kind == "diamond":
        return poly([(x, y - d * 1.15), (x + d * 1.05, y), (x, y + d * 1.15),
                     (x - d * 1.05, y)], fill, stroke, 0.4)
    if kind == "square":
        return rect(x - d * 0.92, y - d * 0.92, d * 1.84, d * 1.84, fill,
                    f' stroke="{stroke}" stroke-width="0.35"')
    if kind == "triangle":
        return poly([(x, y - d * 1.2), (x + d * 1.1, y + d * 0.8),
                     (x - d * 1.1, y + d * 0.8)], fill, stroke, 0.35)
    return circle(x, y, d, fill, stroke, 0.35)


def funnel_panel(p, rows, W, H):
    mL, mR, mT, mB = 17.0, 16.0, 11.0, 15.0
    g = {"x0": mL, "x1": W - mR, "y0": H - mB, "y1": mT, "yTitleX": 6.6}
    plot_w, plot_h = g["x1"] - g["x0"], g["y0"] - g["y1"]
    y_min, y_max = 20.0, 72.0

    def sy(v):
        return g["y0"] - (v - y_min) / (y_max - y_min) * plot_h

    g["sy"] = sy
    ticks = [20, 30, 40, 50, 60, 70]
    present = set(rows["x"])
    cats = [c for c in ORDER if c in present]      # ORDER, not the final share
    band = plot_w / len(cats)
    s = ""
    for t in ticks:
        s += line(g["x0"], sy(t), g["x1"], sy(t), pt=0.5, stroke="#E0E0E0")
    s += axes(g, ticks, p["yTitle"], "int")
    s += line(g["x0"], sy(50), g["x1"], sy(50), pt=0.5, dash="1.2 0.9")
    s += text(g["x1"] + 1.4, sy(50) + SZ["small"] * 0.36, "parity",
              size=SZ["small"], fill=C["soft"])
    n = len(p["stages"])
    offs = [(k / (n - 1) - 0.5) * band * 0.5 for k in range(n)]
    for i, cat in enumerate(cats):
        base = g["x0"] + band * (i + 0.5)
        ys = []
        for st in p["stages"]:
            m = rows[(rows.x == cat) & (rows.s == st[0])]
            ys.append(None if m.empty else float(m.v.iloc[0]))
        if any(v is None for v in ys):
            continue
        for k in range(1, len(ys)):
            s += line(base + offs[k - 1], sy(ys[k - 1]), base + offs[k],
                      sy(ys[k]), pt=0.6, stroke=C["faint"])
        for k, st in enumerate(p["stages"]):
            s += marker(st[2], base + offs[k], sy(ys[k]), st[3], st[4])
        s += text(base, g["y0"] + SZ["tick"] + 1.6, cat, anchor="middle",
                  size=SZ["tick"], fill=C["tick"])
    lx = g["x0"]
    for st in p["stages"]:
        s += marker(st[2], lx + 1.1, 4.2, st[3], st[4])
        s += text(lx + 3, 4.2 + SZ["small"] * 0.36, st[1], size=SZ["small"])
        lx += len(st[1]) * SZ["small"] * 0.56 + 7
    s += text((g["x0"] + g["x1"]) / 2, H - 1.2, p["xTitle"], anchor="middle",
              size=SZ["axis"])
    return s


def bar_panel(p, rows, W, H):
    """Vertical bars: grouped levels, a single signed series, or stacked pairs."""
    mL, mR, mT = 17.0, 27.0, 9.0
    mB = 11.5 if p.get("noX") else 17.0
    g = {"x0": mL, "x1": W - mR, "y0": H - mB, "y1": mT, "yTitleX": 6.6}
    plot_w, plot_h = g["x1"] - g["x0"], g["y0"] - g["y1"]
    y_min, y_max = p["y"]["min"], p["y"]["max"]
    if y_min is None or y_max is None:
        y_min, y_max = pad_range(list(rows.v) + [0])

    def sy(v):
        return g["y0"] - (v - y_min) / (y_max - y_min) * plot_h

    g["sy"] = sy
    if p["y"]["min"] == 0 and p["y"]["max"] == 100:
        ticks = [0, 25, 50, 75, 100]
    elif p["y"]["min"] == -44:
        ticks = [-40, -20, 0, 20, 40]
    else:
        ticks = nice_ticks(y_min, y_max)

    present = list(dict.fromkeys(rows.x))
    cats = [c for c in (p.get("order") or present) if c in present] \
        if p.get("order") else present
    band = plot_w / len(cats)
    s = axes(g, ticks, p.get("yTitle"), p["y"].get("fmt"))

    for rl in p.get("ref", []):
        y = sy(rl["at"])
        s += line(g["x0"], y, g["x1"] + 1.5, y, pt=0.5, dash="1.2 0.9")
        s += text(g["x1"] + 2.4, y + SZ["small"] * 0.36, rl["label"],
                  size=SZ["small"], fill=C["soft"])

    def val(cat, ser):
        # Greece has no business/law figure at the top 10% and top 20%. The row
        # is present but empty, and the prototype keeps the category on the axis
        # marked n/a rather than dropping it, so a missing value is not the same
        # as a missing row: both must come back as None.
        m = rows[(rows.x == cat) & (rows.s == ser)]
        if m.empty:
            return None
        v = float(m.v.iloc[0])
        return None if math.isnan(v) else v

    if p["chart"] == "stacked":
        groups = [("STEM first", "first"), ("Any STEM in list", "any")]
        bw = band * 0.34
        for i, cat in enumerate(cats):
            for k, (gname, short) in enumerate(groups):
                cx = g["x0"] + band * i + band * (0.28 + 0.44 * k)
                f = val(cat, f"female / {gname}")
                m = val(cat, f"male / {gname}")
                if f is None or m is None:
                    continue
                y_f, y_top = sy(f), sy(f + m)
                s += rect(cx - bw / 2, y_f, bw, g["y0"] - y_f, C["female"])
                s += rect(cx - bw / 2, y_top, bw, y_f - y_top, C["male"])
                s += text(cx, y_f + SZ["small"] + 0.8, fmt_val(f, "pct"),
                          anchor="middle", size=SZ["small"])
                s += text(cx, y_top + SZ["small"] + 0.8, fmt_val(m, "pct"),
                          anchor="middle", size=SZ["small"], fill="#FFFFFF")
                s += text(cx, g["y0"] + SZ["small"] + 0.9, short,
                          anchor="middle", size=SZ["small"], fill=C["soft"])
            s += text(g["x0"] + band * (i + 0.5),
                      g["y0"] + SZ["small"] + SZ["tick"] + 1.9, cat,
                      anchor="middle", size=SZ["tick"])
        s += legend_swatch(g["x0"], 4.6, C["female"], "female", SZ["small"])
        s += legend_swatch(g["x0"] + 16, 4.6, C["male"], "male", SZ["small"])
    else:
        sers = p["seriesOrder"]
        bw = band * 0.72 / len(sers)
        for i, cat in enumerate(cats):
            any_val = False
            for k, ser in enumerate(sers):
                v = val(cat, ser)
                if v is None:
                    continue
                any_val = True
                cx = g["x0"] + band * i + band * 0.14 + bw * (k + 0.5)
                y = sy(v)
                zero = sy(max(y_min, min(y_max, 0)))
                fill = (C["male"] if v >= 0 else C["neg"]) \
                    if p["chart"] == "single" else series_fill(ser)
                s += rect(cx - bw * 0.44, min(y, zero), bw * 0.88,
                          abs(zero - y), fill)
                if p["chart"] == "single":
                    s += text(cx, y - 1.1 if v >= 0 else y + SZ["small"] + 0.5,
                              fmt_val(v, "signed"), anchor="middle",
                              size=SZ["small"])
            if not any_val:
                zero = sy(max(y_min, min(y_max, 0)))
                s += text(g["x0"] + band * (i + 0.5), zero - 1.2, "n/a",
                          anchor="middle", size=SZ["small"], fill=C["soft"])
            s += text(g["x0"] + band * (i + 0.5), g["y0"] + SZ["tick"] + 1.6,
                      cat, anchor="middle", size=SZ["tick"])
        if len(sers) > 1:
            lx = g["x0"]
            for ser in sers:
                s += legend_swatch(lx, 4.6, series_fill(ser), ser, SZ["small"])
                lx += len(ser) * SZ["small"] * 0.56 + 6
    if p.get("xTitle") and not p.get("noX"):
        s += text((g["x0"] + g["x1"]) / 2, H - 1.2, p["xTitle"],
                  anchor="middle", size=SZ["axis"])
    if p.get("letter"):
        s += text(0.5, 3.2, p["letter"], size=SZ["letter"], weight="700")
    return s


def place_labels(pts, placed, g, W, plot_w, plot_h):
    """Put a name beside each point: nearest free slot if it is clear and
    unambiguous, otherwise open space reached by a leader that crosses nothing.

    Transcribed from the prototype. `pts` are mutated in place, each gaining a
    `pick` describing where its label goes.
    """
    lh, rad = SZ["small"] * 1.05, 0.95
    leads = []

    def over(b):
        return sum(max(0, min(b["x1"], o["x1"]) - max(b["x0"], o["x0"]))
                   * max(0, min(b["y1"], o["y1"]) - max(b["y0"], o["y0"]))
                   for o in placed)

    def seg_dist(ax, ay, bx, by, px, py):
        dx, dy = bx - ax, by - ay
        L = dx * dx + dy * dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
        return math.hypot(ax + t * dx - px, ay + t * dy - py)

    def seg_cross(a, b, c, d):
        def o(p, q, r):
            v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
            return (v > 0) - (v < 0)
        return o(a, b, c) != o(a, b, d) and o(c, d, a) != o(c, d, b)

    def seg_clear(own, tx, ty):
        for q in pts:
            if q is not own and seg_dist(own["cx"], own["cy"], tx, ty,
                                         q["cx"], q["cy"]) < rad + 0.4:
                return False
        A, B = (own["cx"], own["cy"]), (tx, ty)
        return not any(seg_cross(A, B, L[0], L[1]) for L in leads)

    for p in pts:
        w = len(str(p["s"])) * SZ["small"] * 0.56
        cands = []
        for k in range(3):
            vy = lh * k
            cands += [{"ax": "middle", "dx": 0, "dy": -1.9 - vy},
                      {"ax": "middle", "dx": 0, "dy": 3.1 + vy},
                      {"ax": "start", "dx": 1.9, "dy": SZ["small"] * 0.36 - vy},
                      {"ax": "end", "dx": -1.9, "dy": SZ["small"] * 0.36 + vy}]
            if k:
                cands += [{"ax": "start", "dx": 1.9, "dy": SZ["small"] * 0.36 + vy},
                          {"ax": "end", "dx": -1.9, "dy": SZ["small"] * 0.36 - vy},
                          {"ax": "start", "dx": 1.4, "dy": -1.6 - vy},
                          {"ax": "end", "dx": -1.4, "dy": 2.8 + vy},
                          {"ax": "start", "dx": 1.4, "dy": 2.8 + vy},
                          {"ax": "end", "dx": -1.4, "dy": -1.6 - vy}]
        cands.sort(key=lambda c: abs(c["dx"]) + abs(c["dy"]))

        def box(c):
            lx, ly = p["cx"] + c["dx"], p["cy"] + c["dy"]
            x0 = lx - w / 2 if c["ax"] == "middle" else (lx if c["ax"] == "start" else lx - w)
            return {"x0": x0 - 0.4, "x1": x0 + w + 0.4,
                    "y0": ly - lh * 1.05, "y1": ly + lh * 0.55}

        def in_bounds(b):
            return not (b["x0"] < g["x0"] - 0.5 or b["x1"] > W - 2
                        or b["y0"] < g["y1"] - 2.5 or b["y1"] > g["y0"] + 1)

        def attributable(b):
            mx, my = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2

            def d2(q):
                return (q["cx"] - mx) ** 2 + (q["cy"] - my) ** 2
            own = d2(p)
            return all(q is p or d2(q) > own for q in pts)

        pick = pick_box = None
        for c in cands:
            b = box(c)
            if in_bounds(b) and over(b) <= 0 and attributable(b):
                pick, pick_box = c, b
                break
        for need_attr in (True, False):
            if pick:
                break
            far, cand, cand_box = float("inf"), None, None
            for gx in range(29):
                for gy in range(23):
                    c = {"ax": "middle",
                         "dx": (g["x0"] + plot_w * gx / 28) - p["cx"],
                         "dy": (g["y1"] + plot_h * gy / 22) - p["cy"]}
                    b = box(c)
                    if not in_bounds(b) or over(b) > 0:
                        continue
                    if need_attr and not attributable(b):
                        continue
                    tx, ty = p["cx"] + c["dx"], p["cy"] + c["dy"] - lh * 0.28
                    if not seg_clear(p, tx, ty):
                        continue
                    d = math.hypot(c["dx"], c["dy"])
                    if d < far:
                        far, c["lead"] = d, True
                        cand, cand_box = c, b
            if cand:
                pick, pick_box = cand, cand_box
        if pick is None:
            best = float("inf")
            for c in cands:
                b = box(c)
                if not in_bounds(b):
                    continue
                score = over(b) + (0 if attributable(b) else 40)
                if score < best:
                    best, pick, pick_box = score, c, b
        if pick is None:
            pick = cands[0]
            pick_box = box(pick)
        if pick.get("lead"):
            leads.append(((p["cx"], p["cy"]),
                          (p["cx"] + pick["dx"], p["cy"] + pick["dy"] - lh * 0.28)))
        placed.append(pick_box)
        p["pick"] = pick


def draw_labels(pts):
    lh = SZ["small"] * 1.05
    s = ""
    for q in pts:
        if q["pick"].get("lead"):
            s += line(q["cx"], q["cy"], q["cx"] + q["pick"]["dx"],
                      q["cy"] + q["pick"]["dy"] - lh * 0.28, pt=0.5,
                      stroke=C["soft"])
    for q in pts:
        s += text(q["cx"] + q["pick"]["dx"], q["cy"] + q["pick"]["dy"], q["s"],
                  anchor=q["pick"]["ax"], size=SZ["small"])
    return s


def scatter_panel(p, rows, W, H):
    """One point per setting against a country-level correlate, with a fit."""
    mL, mR, mT = 15.0, 24.0, 8.0
    if p.get("title"):
        mT = 12.0
    mB = 8.5 if p.get("noX") else 14.0
    g = {"x0": mL, "x1": W - mR, "y0": H - mB, "y1": mT, "yTitleX": 6.6}
    plot_w, plot_h = g["x1"] - g["x0"], g["y0"] - g["y1"]
    y_min, y_max = pad_range(list(rows.v) + [0])
    if p["y"].get("min") is not None:
        y_min = p["y"]["min"]
    if p["y"].get("max") is not None:
        y_max = p["y"]["max"]
    x_min, x_max = p["x"]["min"], p["x"]["max"]

    def sy(v):
        return g["y0"] - (v - y_min) / (y_max - y_min) * plot_h

    def sx(v):
        return g["x0"] + (v - x_min) / (x_max - x_min) * plot_w

    g["sy"] = sy
    s = axes(g, nice_ticks(y_min, y_max), p["yTitle"], p["y"].get("fmt"))
    step = p["x"].get("step", 0.1)
    t = x_min
    while t <= x_max + 1e-9:
        s += line(sx(t), g["y0"], sx(t), g["y0"] + 0.8, stroke=C["rule"], pt=0.55)
        if not p.get("noX"):
            s += text(sx(t), g["y0"] + SZ["tick"] + 1.4, f"{t:.1f}",
                      anchor="middle", size=SZ["tick"], fill=C["tick"])
        t += step
    for rl in p.get("ref", []):
        y = sy(rl["at"])
        s += line(g["x0"], y, g["x1"] + 1.2, y, pt=0.5, dash="1.2 0.9")
        s += text(g["x1"] + 2, y + SZ["small"] * 0.36, rl["label"],
                  size=SZ["small"], fill=C["soft"])

    xs, ys = list(rows.xn), list(rows.v)
    n = len(xs)
    sxx, syy = sum(xs), sum(ys)
    sxy = sum(a * b for a, b in zip(xs, ys))
    sx2 = sum(a * a for a in xs)
    slope = (n * sxy - sxx * syy) / (n * sx2 - sxx * sxx)
    icpt = (syy - slope * sxx) / n
    cid = "clip" + re.sub(r"[^A-Za-z0-9]", "", str(p.get("csv", "s")))
    s += (f'<clipPath id="{cid}"><rect x="{r3(g["x0"])}" y="{r3(g["y1"])}" '
          f'width="{r3(plot_w)}" height="{r3(plot_h)}" /></clipPath>')
    s += (f'<g clip-path="url(#{cid})">'
          + line(sx(x_min), sy(slope * x_min + icpt),
                 sx(x_max), sy(slope * x_max + icpt), pt=0.7, stroke=C["rule"])
          + "</g>")

    pts = sorted(({"cx": sx(r.xn), "cy": sy(r.v), "s": r.s, "v": r.v}
                  for r in rows.itertuples()), key=lambda q: q["cx"])
    lh, rad, pad = SZ["small"] * 1.05, 0.95, 0.3
    placed = [{"x0": q["cx"] - rad - pad, "x1": q["cx"] + rad + pad,
               "y0": q["cy"] - rad - pad, "y1": q["cy"] + rad + pad} for q in pts]
    for rl in p.get("ref", []):
        if p.get("title"):
            # Keep the promoted figure's labels clear of the zero line.
            placed.append({"x0": g["x0"], "x1": g["x1"],
                           "y0": sy(rl["at"]) - 0.4,
                           "y1": sy(rl["at"]) + 0.4})
        ly = sy(rl["at"]) + SZ["small"] * 0.36
        lw = len(rl["label"]) * SZ["small"] * 0.56
        placed.append({"x0": g["x1"] + 1.6, "x1": g["x1"] + 2.4 + lw,
                       "y0": ly - lh * 1.05, "y1": ly + lh * 0.55})
    for k in range(61):
        xv = x_min + (x_max - x_min) * k / 60
        yv = slope * xv + icpt
        if yv < y_min or yv > y_max:
            continue
        lx, ly = sx(xv), sy(yv)
        placed.append({"x0": lx - 0.9, "x1": lx + 0.9,
                       "y0": ly - 0.55, "y1": ly + 0.55})
    place_labels(pts, placed, g, W, plot_w, plot_h)
    # The corrected Taiwan parity places Taiwan, Chile and Finland almost on
    # top of each other. Separate their labels with explicit leaders in this
    # panel; do not move the observations or the fitted line.
    if p.get("csv") == "fig4_b":
        offsets = {"Finland": (-6, -8, "end"), "Chile": (0, -12, "middle"),
                   "Taiwan": (5, 5, "start")}
        for q in pts:
            if q["s"] in offsets:
                dx, dy, anchor = offsets[q["s"]]
                q["pick"] = {"dx": dx, "dy": dy, "ax": anchor, "lead": True}
    s += draw_labels(pts)
    for q in pts:
        s += circle(q["cx"], q["cy"], 1.15,
                    C["male"] if q["v"] >= 0 else C["neg"], "#FFFFFF", 0.3)
    if not p.get("noX"):
        s += text((g["x0"] + g["x1"]) / 2, H - 1.2, p["xTitle"],
                  anchor="middle", size=SZ["axis"])
    if p.get("letter"):
        s += text(0.5, 3.2, p["letter"], size=SZ["letter"], weight="700")
    if p.get("title"):
        s += text(mL, 4.0, p["title"], size=SZ["letter"], weight="700")
    return s


def smallmult_panel(p, rows, W, H):
    """One mini panel per country, sharing a scale; bars are score bins."""
    cols, rows_n = 5, 2
    mL, mR, mT, mB = 15.0, 3.0, 4.0, 13.0
    gw, gh = (W - mL - mR) / cols, (H - mT - mB) / rows_n
    cats = p["order"]
    y_min, y_max = p["y"]["min"], p["y"]["max"]
    ticks = [0, 50, 100] if (y_min == 0 and y_max == 100) else [-40, 0, 40]
    names = list(dict.fromkeys(rows.s))

    def resolve(c):
        if c in names:
            return c
        for n in names:
            if str(n).startswith(c):
                return n
        return c

    s = ""
    for idx, country in enumerate(resolve(c) for c in p["panelOrder"]):
        ci, ri = idx % cols, idx // cols
        ox, oy = mL + ci * gw, mT + ri * gh
        px0, px1 = ox + 1, ox + gw - 2.5
        py1, py0 = oy + 5, oy + gh - 8

        def sy(v, _py0=py0, _py1=py1):
            return _py0 - (v - y_min) / (y_max - y_min) * (_py0 - _py1)

        s += text(px0, oy + 3.2, country, size=SZ["small"], weight="600")
        s += line(px0, py1, px0, py0, pt=0.55, stroke=C["rule"])
        s += line(px0, py0, px1, py0, pt=0.55, stroke=C["rule"])
        for t in ticks:
            s += line(px0 - 0.7, sy(t), px0, sy(t), pt=0.55, stroke=C["rule"])
            if ci == 0:
                s += text(px0 - 1.2, sy(t) + SZ["small"] * 0.36,
                          fmt_val(t, p["y"].get("fmt")), anchor="end",
                          size=SZ["small"], fill=C["tick"])
        for rl in p.get("ref", []):
            s += line(px0, sy(rl["at"]), px1, sy(rl["at"]), pt=0.5, dash="1 0.8")
        band = (px1 - px0) / len(cats)
        for i, cat in enumerate(cats):
            m = rows[(rows.x == cat) & (rows.s == country)]
            if m.empty:
                continue
            v = float(m.v.iloc[0])
            cx = px0 + band * (i + 0.5)
            y = sy(v)
            zero = sy(max(y_min, min(y_max, 0)))
            fill = C["male"] if y_min == 0 else (C["male"] if v >= 0 else C["neg"])
            s += rect(cx - band * 0.32, min(y, zero), band * 0.64,
                      abs(zero - y), fill)
            if ri == rows_n - 1:
                s += text(cx, py0 + SZ["small"] + 1.1, cat, anchor="middle",
                          size=SZ["small"], fill=C["tick"])
    s += text(4.6, H / 2, p["yTitle"], anchor="middle", size=SZ["axis"],
              rotate=-90)
    s += text(W / 2, H - 1.2, p["xTitle"], anchor="middle", size=SZ["axis"])
    return s


# ------------------------------------------------------------------ assembly
def svg_doc(body: str, w: float, h: float) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{r3(w)}mm" '
            f'height="{r3(h)}mm" viewBox="0 0 {r3(w)} {r3(h)}">'
            f'<rect width="{r3(w)}" height="{r3(h)}" fill="#FFFFFF" />'
            f'{body}</svg>')


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}.csv")
    ren = {"x": "x", "series": "s", "y": "v"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    # The decomposition panel carries the numeric P in the x column and the
    # setting name in series, the reverse of the categorical panels.
    if "xn" not in df.columns and pd.api.types.is_numeric_dtype(df["x"]):
        df["xn"] = df["x"].astype(float)
    return df


def write(name: str, svg: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.svg"
    p.write_text(svg, encoding="utf-8")
    try:
        # invariant must be set before reportlab builds the document, or it
        # stamps a creation time and a random document id and every rebuild
        # shows sixteen modified binaries in a repo that tracks them.
        from reportlab import rl_config
        rl_config.invariant = 1
        if name == "fig5_wage_parity":
            # Embed the bundled sans-serif face so journal/PDF viewers do not
            # substitute a font with different glyph widths in this new plot.
            import reportlab
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            fonts = Path(reportlab.__file__).parent / "fonts"
            pdfmetrics.registerFont(TTFont("Vera", str(fonts / "Vera.ttf")))
            pdfmetrics.registerFont(TTFont("Vera-Bold", str(fonts / "VeraBd.ttf")))
            pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="Vera-Bold")
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        # Stage on the destination filesystem so the final replacement is
        # atomic. Rendering or embedding failures leave the existing PDF intact.
        with tempfile.TemporaryDirectory(dir=OUT, prefix=f".{name}-") as directory:
            staged = Path(directory) / p.with_suffix(".pdf").name
            renderPDF.drawToFile(svg2rlg(str(p)), str(staged))
            embed_fonts(staged)
            staged.replace(p.with_suffix(".pdf"))
    except Exception as e:                                   # pragma: no cover
        print(f"  ! {name}: SVG written, PDF export failed ({e})")
        raise
    return p


def embed_fonts(path: Path) -> None:
    """Keep vector geometry while embedding base-14 fonts for portable labels.

    Some PDF viewers substitute a face with different glyph widths when the
    gallery's Helvetica is not embedded. Ghostscript supplies the compatible
    font without rebuilding points, lines, or labels from data. Existing PDFs
    can also be repaired by calling this function directly.
    """
    gs = shutil.which("gs") or shutil.which("gswin64c")
    if gs is None:
        raise RuntimeError("Ghostscript is required to embed figure fonts")
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "embedded.pdf"
        subprocess.run([
            gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
            "-dEmbedAllFonts=true", "-dSubsetFonts=true",
            "-dOmitInfoDate=true", "-dOmitID=true",
            f"-sOutputFile={target}",
            "-c", "<</NeverEmbed []>> setdistillerparams", "-f", str(path),
        ], check=True)
        path.write_bytes(target.read_bytes())


# --------------------------------------------------------------- the figures
def figure1() -> str:
    # W_DOUBLE minus the 13mm of dead left margin and 12mm of gutter the panel
    # no longer reserves, so the plot area is unchanged while the change column
    # sits 12mm closer to the axis and the canvas hugs its ink.
    W, H = W_DOUBLE - 25.0, 83.0
    p = {"chart": "dumbbell",
         "xTitle": "Women as a share of everyone who ranks a STEM program first (%)",
         "pairs": [["top 10% of scorers", "top 10% of scorers"],
                   ["all applicants", "all applicants"]],
         "x": {"min": 0, "max": 60},
         "ref": [{"at": 50, "label": "parity"}]}
    return svg_doc(dumbbell_panel(p, load("fig1_revised"), W, H), W, H)


def figure2() -> str:
    W, H = W_DOUBLE, 78.0
    half = W / 2
    a = {"chart": "hbar", "letter": "a", "paired": True, "gutter": 34.0,
         "span": half - 46.0, "colX": half - 8.0, "vTitleX": half - 3.0,
         "colTitle": "Women's share of top scorers (%)",
         "xTitle": "Pipeline gap (female - male, pp)", "xr": (-35, 35)}
    b = dict(a, letter="b", colTitle="Top-10% women ranking STEM first (%)",
             xTitle="Choice gap (female - male, pp)", xr=(-44, 20),
             noNames=True)
    body = hbar_panel(a, load("fig2_b"), W, H, load("fig2_a"))
    right = hbar_panel(b, load("fig3_b"), W, H, load("fig3_a"))
    body += f'<g transform="translate({r3(half)} 0)">{right}</g>'
    return svg_doc(body, W, H)


def figure3() -> str:
    W, H = 140.0, 92.0
    p = {"chart": "decomp",
         "xTitle": sub_title("Pipeline component P = ", ("T", "m"), ("T", "f"),
                             " (log points)"),
         "yTitle": sub_title("Choice component C = ", ("s", "m"), ("s", "f"),
                             " (log points)")}
    return svg_doc(decomp_panel(p, load("fig6"), W, H), W, H)


def figure4() -> str:
    W, H = W_DOUBLE, 88.0
    p = {"chart": "funnel", "xTitle": "Country",
         "yTitle": "Female share within each group (%)",
         "stages": [["In top 10%", "in top 10%", "diamond", "#FFFFFF", "#2A6F6B"],
                    ["Ranked STEM anywhere", "ranked STEM anywhere", "circle",
                     "#A9C6C3", "#1F4F4B"],
                    ["Ranked STEM first", "ranked STEM first", "triangle",
                     "#5E938E", "#1F4F4B"],
                    ["Assigned to STEM", "assigned to STEM", "square",
                     "#1F4F4B", "#1F4F4B"]]}
    return svg_doc(funnel_panel(p, load("figR0"), W, H), W, H)


def figure5() -> str:
    """Promoted wage-parity plot: unchanged points, common percentage-point scale.

    Native six-inch width keeps country labels legible in the manuscript.
    Both fitted lines are descriptive, unweighted OLS; sensitivity is in
    Appendix G. The existing appendix rendering remains reproducible below.
    """
    panels = []
    for csv, title in [("fig4_a", "a  Academic pipeline"),
                       ("fig4_b", "b  First-choice STEM applications")]:
        panels.append({"csv": csv, "chart": "scatter", "title": title,
                       "xTitle": "Wage parity index (1 - gender wage gap)",
                       "yTitle": "Female minus male (percentage points)",
                       "x": {"min": 0.5, "max": 1.0},
                       "y": {"min": -40, "max": 40, "fmt": "signed"},
                       "ref": [{"at": 0, "label": "no gap"}]})
    return stack_panels(panels, width=152.4, base_h=82.0).replace(
        'font-family="Helvetica"', 'font-family="Vera"')


# ------------------------------------------------------- appendix figures
LEVELS = {"min": 0, "max": 100, "fmt": "pct"}
GAPS = {"min": -44, "max": 44, "fmt": "signed"}
YT = {"share": "Share among top score group (%)",
      "prob": "Probability within top score group (%)",
      "gapPP": "Female minus male (pp)",
      "shares": "Gender shares among STEM applicants (%)"}
BINS = ["1-5", "6", "7", "8", "9", "10"]


def lv(csv, letter, y_title):
    return {"csv": csv, "letter": letter, "chart": "grouped", "yTitle": y_title,
            "xTitle": "Country", "order": ORDER, "seriesOrder": ["male", "female"],
            "y": LEVELS, "ref": []}


def gp(csv, letter):
    return {"csv": csv, "letter": letter, "chart": "single", "yTitle": YT["gapPP"],
            "xTitle": "Country", "order": ORDER,
            "seriesOrder": ["female minus male"], "y": GAPS,
            "ref": [{"at": 0, "label": "no gender gap"}]}


def st(csv, letter):
    return {"csv": csv, "letter": letter, "chart": "stacked",
            "yTitle": YT["shares"], "xTitle": "Country", "order": ORDER,
            "y": LEVELS, "ref": [{"at": 50, "label": "parity"}]}


def sm(csv, y_title, y, ref):
    return {"csv": csv, "letter": None, "chart": "smallmult", "yTitle": y_title,
            "xTitle": "Score bin (1-5 pooled, then deciles 6-10)",
            "order": BINS, "panelOrder": ORDER, "y": y, "ref": ref}


def stack_panels(panels, width=W_DOUBLE, base_h=None):
    """Lay panels out in one column, the prototype's single-column case.

    Every panel but the last is drawn with no x labels and gives the 5.5mm
    axis-title band back to the figure, exactly as the prototype does.
    """
    kind = panels[0]["chart"]
    base_h = base_h or (118.0 if kind == "smallmult"
                        else 74.0 if kind == "scatter" else H_BAR)
    n = len(panels)
    heights = [base_h if i == n - 1 else base_h - 5.5 for i in range(n)]
    draw = {"smallmult": smallmult_panel, "scatter": scatter_panel}
    body, y = "", 0.0
    for p, h in zip(panels, heights):
        q = dict(p, noX=(h != base_h))
        rows = load(p["csv"])
        fn = draw.get(p["chart"], bar_panel)
        body += f'<g transform="translate(0 {r3(y)})">{fn(q, rows, width, h)}</g>'
        y += h
    return svg_doc(body, width, y)


APPENDIX = {
    "figB1_shares_thresholds": lambda: stack_panels(
        [st("figB1_a", "a"), st("figB1_b", "b")]),
    "figB2_pipeline_thresholds": lambda: stack_panels(
        [lv("figB2_a", "a", YT["share"]), gp("figB2_b", "b"),
         lv("figB2_c", "c", YT["share"]), gp("figB2_d", "d")]),
    "figB3_stem_first_thresholds": lambda: stack_panels(
        [lv("figB3_a", "a", YT["prob"]), gp("figB3_b", "b"),
         lv("figB3_c", "c", YT["prob"]), gp("figB3_d", "d")]),
    "figB4_stem_any_thresholds": lambda: stack_panels(
        [lv("figB4_a", "a", YT["prob"]), gp("figB4_b", "b"),
         lv("figB4_c", "c", YT["prob"]), gp("figB4_d", "d"),
         lv("figB4_e", "e", YT["prob"]), gp("figB4_f", "f")]),
    # figB4, figC1 and figC3 stack six panels. At the width of the text block
    # that is 13in tall -- taller than the page -- so keepaspectratio squeezed
    # them to half width and took the labels down with them. Splitting levels
    # from gaps puts three panels on a page at full width, and the reader
    # compares thresholds down one column instead of skipping every other panel.
    "figB2_pipeline_levels": lambda: stack_panels(
        [lv("figB2_a", "a", YT["share"]), lv("figB2_c", "b", YT["share"])]),
    "figB2_pipeline_gaps": lambda: stack_panels(
        [gp("figB2_b", "a"), gp("figB2_d", "b")]),
    "figB3_stem_first_levels": lambda: stack_panels(
        [lv("figB3_a", "a", YT["prob"]), lv("figB3_c", "b", YT["prob"])]),
    "figB3_stem_first_gaps": lambda: stack_panels(
        [gp("figB3_b", "a"), gp("figB3_d", "b")]),
    "figB4_stem_any_levels": lambda: stack_panels(
        [lv("figB4_a", "a", YT["prob"]), lv("figB4_c", "b", YT["prob"]),
         lv("figB4_e", "c", YT["prob"])]),
    "figB4_stem_any_gaps": lambda: stack_panels(
        [gp("figB4_b", "a"), gp("figB4_d", "b"), gp("figB4_f", "c")]),
    "figB5_female_shares_bins": lambda: stack_panels(
        [sm("figB5", "Female share (%)", LEVELS,
            [{"at": 50, "label": "parity"}])]),
    "figB6_stem_first_bins": lambda: stack_panels(
        [sm("figB6", YT["gapPP"], GAPS, [{"at": 0, "label": "no gap"}])]),
    "figB7_stem_any_bins": lambda: stack_panels(
        [sm("figB7", YT["gapPP"], GAPS, [{"at": 0, "label": "no gap"}])]),
    # One grouped panel rather than three stacked ones: the point is a
    # within-country comparison across thresholds, which the reader makes by
    # looking at three adjacent bars, not by tracking a country down a column.
    "figB8_thresholds": lambda: stack_panels(
        [{"csv": "figB8_thresholds", "letter": None, "chart": "grouped",
          "yTitle": YT["gapPP"], "xTitle": "Country", "order": ORDER,
          "seriesOrder": ["Top 1%", "Top 5%", "Top 10%"], "y": GAPS,
          "ref": [{"at": 0, "label": "no gender gap"}]}]),
    "figC1_business_first": lambda: stack_panels(
        [lv("figC1_a", "a", YT["prob"]), gp("figC1_b", "b"),
         lv("figC1_c", "c", YT["prob"]), gp("figC1_d", "d"),
         lv("figC1_e", "e", YT["prob"]), gp("figC1_f", "f")]),
    "figC2_business_any": lambda: stack_panels(
        [lv("figC2_a", "a", YT["prob"]), gp("figC2_b", "b"),
         lv("figC2_c", "c", YT["prob"]), gp("figC2_d", "d"),
         lv("figC2_e", "e", YT["prob"]), gp("figC2_f", "f")]),
    "figC3_health_first": lambda: stack_panels(
        [lv("figC3_a", "a", YT["prob"]), gp("figC3_b", "b"),
         lv("figC3_c", "c", YT["prob"]), gp("figC3_d", "d"),
         lv("figC3_e", "e", YT["prob"]), gp("figC3_f", "f")]),
    "figC1_business_first_levels": lambda: stack_panels(
        [lv("figC1_a", "a", YT["prob"]), lv("figC1_c", "b", YT["prob"]),
         lv("figC1_e", "c", YT["prob"])]),
    "figC1_business_first_gaps": lambda: stack_panels(
        [gp("figC1_b", "a"), gp("figC1_d", "b"), gp("figC1_f", "c")]),
    "figC3_health_first_levels": lambda: stack_panels(
        [lv("figC3_a", "a", YT["prob"]), lv("figC3_c", "b", YT["prob"]),
         lv("figC3_e", "c", YT["prob"])]),
    "figC3_health_first_gaps": lambda: stack_panels(
        [gp("figC3_b", "a"), gp("figC3_d", "b"), gp("figC3_f", "c")]),
    "figC4_health_any": lambda: stack_panels(
        [lv("figC4_a", "a", YT["prob"]), gp("figC4_b", "b"),
         lv("figC4_c", "c", YT["prob"]), gp("figC4_d", "d"),
         lv("figC4_e", "e", YT["prob"]), gp("figC4_f", "f")]),
    # Wage parity only. The prototype pairs these with GDP per capita, but the
    # editor's comment 6 asked us to move away from GDP as the development
    # proxy and the letter's answer to comment 7 says we replaced it, so the
    # GDP column is deliberately not built.
    "figH1_wage_parity": lambda: stack_panels(
        [{"csv": "fig4_a", "letter": "a", "chart": "scatter",
          "xTitle": "Wage parity index (1 - gender wage gap)",
          "yTitle": "Pipeline gap (female - male, pp)",
          "x": {"min": 0.5, "max": 1.0},
          "y": {"min": None, "max": None, "fmt": "signed"},
          "ref": [{"at": 0, "label": "no gap"}]},
         {"csv": "fig4_b", "letter": "b", "chart": "scatter",
          "xTitle": "Wage parity index (1 - gender wage gap)",
          "yTitle": "STEM first-choice gap (female - male, pp)",
          "x": {"min": 0.5, "max": 1.0},
          "y": {"min": None, "max": None, "fmt": "signed"},
          "ref": [{"at": 0, "label": "no gap"}]}]),
    "figD1_recent_cohort": lambda: stack_panels(
        [{"csv": "figD1", "letter": None, "chart": "grouped",
          "yTitle": YT["gapPP"], "xTitle": "Country", "order": ORDER,
          "seriesOrder": ["pipeline gap", "STEM first-choice gap",
                          "any-STEM gap"],
          "y": {"min": None, "max": None, "fmt": "signed"},
          "ref": [{"at": 0, "label": "no gender gap"}]}]),
}

MAIN = {
    "fig1_stem_share": figure1,
    "fig2_pipeline_choice": figure2,
    "fig3_decomposition_scatter": figure3,
    "fig4_stem_funnel": figure4,
    "fig5_wage_parity": figure5,
}


def main():
    ok = fail = 0
    for name, fn in list(MAIN.items()) + list(APPENDIX.items()):
        try:
            write(name, fn())
            print(f"  wrote {name}")
            ok += 1
        except Exception as e:
            print(f"  ! {name} FAILED: {type(e).__name__}: {e}")
            fail += 1
    print(f"\n{ok} figures written, {fail} failed")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

