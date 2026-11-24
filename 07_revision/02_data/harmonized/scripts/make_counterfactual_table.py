#!/usr/bin/env python3
"""Counterfactual female shares among high-achieving students assigned to STEM.

Both counterfactuals are readings of the identity in
sections/appendix/E_decomposition.tex, each holding one margin fixed and neutralising
the other. With T_g the top-decile count and M_g the count assigned to STEM:

    observed            M_f / (M_f + M_m)
    balanced pipeline   a_f / (a_f + a_m),  a_g = M_g / T_g   [= 1 - L(C + V)]
    common STEM assignment rates T_f / (T_f + T_m)

The first neutralises the pipeline and leaves the choice and assignment margins
as observed; the second neutralises those margins and leaves the pipeline as
observed. Neither is a behavioural prediction: they reallocate the observed
rates, and say nothing about how either margin would respond to an intervention.

Outputs:
- figures/table_counterfactuals.tex
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path

# Printed setting names. The harmonized data keep the key "China"; the table
# prints the coauthors' neutral label for the Ningxia data (November 2026).
DISPLAY = {"China": "Mainland China"}


def tex_num(x, spec):
    """Format a number for a LaTeX table cell with a real minus sign.

    An ASCII hyphen in text mode prints as a hyphen, which is visibly shorter
    than the minus the prose sets in math mode; $-$ gives the same glyph.
    """
    s = format(x, spec)
    return s.replace("-", "$-$", 1) if s.startswith("-") else s

BASE = Path(__file__).resolve().parent.parent
MASTER = BASE / "master" / "harmonized_counts_long.csv"
OUT = BASE / "figures" / "table_counterfactuals.tex"

# Authoritative copy lives in create_editor_comment_figures.py; keep in step.
# A stale name here filters a setting to zero rows, so validate() checks it.
PREFERRED = {
    "Australia": "atar_pooled_2009_2010", "Brazil": "reply_tables",
    "Chile": "chile_2012_2018", "China": "china_ningxia_2018",
    "Finland": "main_pooled", "Greece": "years_2003_2012",
    "Spain": "esp_2020_2024_tables", "Sweden": "reply_docx_2017",
    "Taiwan": "reply_pdf_2010", "Uganda": "reply_docx",
}
GROUP = "Decile 10"


def load():
    counts = collections.defaultdict(float)
    seen = collections.defaultdict(set)
    for row in csv.DictReader(MASTER.open()):
        if row["score_group_label"] != GROUP:
            continue
        seen[row["country"]].add(row["sample_variant"])
        if PREFERRED.get(row["country"]) != row["sample_variant"]:
            continue
        counts[(row["country"], row["outcome"], row["sex"])] += float(row["n"])
    return counts, seen


def validate(seen):
    """Fail loudly on a stale variant instead of silently dropping a setting."""
    stale = {c: v for c, v in PREFERRED.items() if v not in seen.get(c, set())}
    if stale:
        raise SystemExit(
            "PREFERRED_VARIANT is stale for: "
            + "; ".join(f"{c} names {v!r}, master has {sorted(seen.get(c, []))}"
                        for c, v in stale.items())
        )


def main():
    counts, seen = load()
    validate(seen)

    rows = []
    for country in sorted(PREFERRED):
        tf = counts.get((country, "pipeline_total", "female"), 0.0)
        tm = counts.get((country, "pipeline_total", "male"), 0.0)
        mf = counts.get((country, "stem_admitted", "female"), 0.0)
        mm = counts.get((country, "stem_admitted", "male"), 0.0)
        if not (tf and tm and mf and mm):
            continue
        observed = 100 * mf / (mf + mm)
        af, am = mf / tf, mm / tm
        balanced = 100 * af / (af + am)
        common = 100 * tf / (tf + tm)
        rows.append((country, observed, balanced, common))

    rows.sort(key=lambda r: r[1])
    falls = [r[0] for r in rows if r[2] < r[1]]

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Counterfactual female shares among high-achieving students "
        r"assigned to STEM}",
        r"\label{tab:counterfactuals}",
        # Sized to fit elsarticle's 345pt textwidth, as in the cross-field table.
        r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r" & & \multicolumn{2}{c}{Counterfactual} & Change from \\",
        r"\cmidrule(lr){3-4}",
        r"Setting & Observed & Balanced pipeline & \shortstack{Common STEM\\assignment rates} "
        r"& balancing \\",
        r"\midrule",
    ]
    for country, observed, balanced, common in rows:
        lines.append(
            f"{DISPLAY.get(country, country)} & {observed:.1f} & {balanced:.1f} & {common:.1f} & "
            f"{tex_num(balanced - observed, '+.1f')} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{\textwidth}\vspace{4pt}\scriptsize \textit{Notes:} "
        r"All entries are the percentage of high-achieving students assigned to a "
        r"STEM program who are women, where high-achieving means the top decile of "
        r"each setting's academic performance distribution. \textit{Observed} is the "
        r"realized share. \textit{Balanced pipeline} holds each gender's rate of "
        r"reaching a STEM assignment at its observed value and sets the top decile to "
        r"equal numbers of women and men; it equals $1-\Lambda(C+V)$ in the notation of "
        r"\eqref{eq:decomposition_assign} (Online \ref{sec:appendix_decomposition}). "
        r"\textit{Common STEM assignment rates} holds the observed top-decile "
        r"composition and gives women men's assignment rate $M_m/T_m$, setting "
        r"$C+V=0$; the female share is then $1-\Lambda(P)$. "
        r"The final column is the balanced-pipeline "
        r"share minus the observed share; a negative entry means that balancing the "
        r"pipeline, on its own, would lower the female share among STEM assignees, "
        r"because women already outnumber men in that setting's top decile. Both "
        r"columns are accounting exercises that reallocate observed rates and are not "
        r"predictions of how either margin would respond to an intervention."
        r"\end{minipage}", r"\end{table}", "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}\n")
    print(f"{'Setting':10s}{'observed':>10s}{'balanced':>10s}{'common':>9s}{'change':>9s}")
    for country, observed, balanced, common in rows:
        print(f"{country:10s}{observed:9.1f}%{balanced:9.1f}%{common:8.1f}%{balanced-observed:+9.1f}")
    print(f"\nbalancing the pipeline lowers the female share in {len(falls)}/{len(rows)}: {falls}")
    print(f"common STEM assignment rates raises it in "
          f"{sum(c > o for _, o, _, c in rows)}/{len(rows)} settings")


if __name__ == "__main__":
    main()

