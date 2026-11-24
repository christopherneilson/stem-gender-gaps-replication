"""Build the log-odds decomposition table (pipeline, choice and assignment, by field).

Identity, at a fixed performance threshold:
    ln(A_m / A_f) = ln(T_m / T_f) + ln(s_m / s_f) = P + C
where T_g is the count of gender g in the top group and s_g the share of them
applying to the field. P is common to all fields within a country by construction.

Extended one stage further, to realized assignments:
    ln(M_m / M_f) = P + C + ln(v_m / v_f) = P + C + V
where M_g is the count assigned to the field and v_g = M_g / A_g. Note v_g is a
ratio of counts, not a conditional probability: students may be assigned to a
field they did not rank first, so v_g can exceed one.
"""
from __future__ import annotations
import csv, math, collections, re
from pathlib import Path


def tex_num(x, spec):
    """Format a number for a LaTeX table cell with a real minus sign.

    An ASCII hyphen in text mode prints as a hyphen, which is visibly shorter
    than the minus the prose sets in math mode; $-$ gives the same glyph.
    """
    s = format(x, spec)
    return s.replace("-", "$-$", 1) if s.startswith("-") else s

BASE = Path(__file__).resolve().parent.parent
MASTER = BASE / "master" / "harmonized_counts_long.csv"
OUT = BASE / "figures" / "table_crossfield_decomposition.tex"
FIGURE_SCRIPT = BASE / "scripts" / "create_editor_comment_figures.py"


def preferred_variants():
    """Read PREFERRED_VARIANT from the figure script rather than duplicating it.

    Variant names get renamed as country files are updated (Australia moved from
    `reply_docx` to `atar_pooled_2009_2010`). A duplicated copy of this mapping
    drifts silently: the country either drops out of the table or, worse, is
    computed from a superseded extract while everything still looks fine.
    """
    block = re.search(
        r"PREFERRED_VARIANT\s*=\s*\{(.*?)\}", FIGURE_SCRIPT.read_text(encoding="utf-8"), re.S
    )
    if not block:
        raise SystemExit(f"Could not find PREFERRED_VARIANT in {FIGURE_SCRIPT}")
    pairs = re.findall(r"[\"']([^\"']+)[\"']\s*:\s*[\"']([^\"']+)[\"']", block.group(1))
    if not pairs:
        raise SystemExit(f"PREFERRED_VARIANT in {FIGURE_SCRIPT} parsed as empty")
    return dict(pairs)


PREFERRED = preferred_variants()
GROUP = "Decile 10"          # top 10%
# (first-preference outcome, assigned outcome, label)
FIELDS = [
    ("stem_first", "stem_admitted", "STEM"),
    ("health_first", "health_admitted", "Health"),
    ("business_first", "business_admitted", "Business/Law"),
]
LABELS = [label for _, _, label in FIELDS]

logistic = lambda z: 1.0 / (1.0 + math.exp(-z))


def load():
    counts = collections.defaultdict(float)
    for row in csv.DictReader(MASTER.open()):
        if row["score_group_label"] != GROUP:
            continue
        if PREFERRED.get(row["country"]) != row["sample_variant"]:
            continue
        counts[(row["country"], row["outcome"], row["sex"])] += float(row["n"])
    return counts


def main():
    counts = load()
    rows = []
    for country in sorted(PREFERRED):
        tf = counts.get((country, "pipeline_total", "female"), 0.0)
        tm = counts.get((country, "pipeline_total", "male"), 0.0)
        if not (tf and tm):
            continue
        pipeline = math.log(tm / tf)
        choice, assign = {}, {}
        for first, admitted, label in FIELDS:
            af = counts.get((country, first, "female"), 0.0)
            am = counts.get((country, first, "male"), 0.0)
            choice[label] = math.log((am / tm) / (af / tf)) if af and am else None
            mf = counts.get((country, admitted, "female"), 0.0)
            mm = counts.get((country, admitted, "male"), 0.0)
            # v_g = M_g / A_g is a ratio of counts and may exceed one.
            assign[label] = math.log((mm / am) / (mf / af)) if af and am and mf and mm else None
        rows.append((country, pipeline, choice, assign))

    def fmt(x):
        if x is None:
            return "--"
        if abs(x) < 5e-3:   # avoid rendering a negative zero as "-0.00"
            x = 0.0
        return tex_num(x, "+.2f")

    def tally(key, comp):
        """'k/n' over settings where the component is defined."""
        vals = [c[key] if comp == "C" else a[key] for _, _, c, a in rows]
        vals = [v for v in vals if v is not None]
        return f"{sum(v > 0 for v in vals)}/{len(vals)}"

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Decomposition of the gender gap among high-achieving applicants "
        r"and assignees, by field}",
        r"\label{tab:crossfield_decomposition}",
        # Sized to fit elsarticle's 345pt textwidth: at default 10pt/6pt the body
        # overruns the text block by ~97pt. Scoped to the float; the notes set their own size.
        r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r" & Pipeline & \multicolumn{3}{c}{Choice component $C$} & "
        r"& Assignment & Balanced-pipeline \\",
        r"\cmidrule(lr){3-5}",
        r"Setting & $P$ & STEM & Health & Business/Law & & $V_{\text{STEM}}$ "
        r"& male share $\Lambda(C_{\text{STEM}})$ \\",
        r"\midrule",
    ]
    for country, pipeline, choice, assign in rows:
        stem = choice["STEM"]
        lines.append(
            f"{country} & {fmt(pipeline)} & {fmt(stem)} & {fmt(choice['Health'])} & "
            f"{fmt(choice['Business/Law'])} & & {fmt(assign['STEM'])} & "
            f"{logistic(stem):.2f} \\\\"
        )
    stems = [c["STEM"] for _, _, c, _ in rows]
    v_stems = [a["STEM"] for _, _, _, a in rows if a["STEM"] is not None]

    def vrange(label):
        vals = [a[label] for _, _, _, a in rows if a[label] is not None]
        return f"$[{min(vals):+.2f},\\, {max(vals):+.2f}]$"

    # Only V_STEM is tabulated, for width. The other fields go to a snippet that
    # Appendix E inputs, so Table 2's note can point at E instead of carrying them,
    # and the numbers stay generated rather than transcribed.
    # Uganda's business/law V is larger in magnitude than any STEM V, so report it
    # separately rather than let it set the range for the other eight settings.
    biz = {country: a["Business/Law"] for country, _, _, a in rows
           if a["Business/Law"] is not None}
    ug_biz = biz.pop("Uganda")
    biz_range = f"$[{min(biz.values()):+.2f},\\, {max(biz.values()):+.2f}]$"
    (OUT.parent / "vrange_other_fields.tex").write_text(
        r"The assignment component is also small in the other fields, ranging over "
        + vrange("Health") + r" for health and, outside Uganda ($" + f"{ug_biz:+.2f}"
        + r"$), over " + biz_range + r" for business/law." + "\n",
        encoding="utf-8", newline="\n")
    lines += [
        r"\midrule",
        f"Positive in & {sum(p > 0 for _, p, _, _ in rows)}/{len(rows)} & "
        + " & ".join(tally(k, "C") for k in LABELS)
        + f" & & {tally('STEM', 'V')} & " + r"\\",
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{\textwidth}\vspace{4pt}\scriptsize \textit{Notes:} "
        r"High-achieving students are those in the top decile of each setting's academic "
        r"performance distribution. $P$ is the pipeline component and $C$ the choice "
        r"component of \eqref{eq:decomposition}; $V$ is the assignment component of "
        r"\eqref{eq:decomposition_assign}, which extends the identity to realized "
        r"assignments. Positive values "
        r"favor men, and $P$ is common to all fields within a setting by construction. "
        r"$\Lambda(C)$ is the male share among high-achieving STEM applicants that would "
        r"obtain if the pipeline were balanced. Online \ref{sec:appendix_decomposition} "
        r"extends the identity to assignments and reports $V$ for the other fields. Greece "
        r"does not report business/law counts at the top decile. Rates here use each "
        r"setting's full top-decile totals as denominators, so that the components satisfy "
        r"the identity exactly; panel (b) of Figure \ref{fig:editor_fig2_pipeline} instead "
        r"conditions Chile's rates on students who submit an application, so Chile's $C$ "
        r"and its plotted gap are not directly comparable. "
        r"\end{minipage}", r"\end{table}", "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    print(f"\nSTEM C: positive in {sum(s > 0 for s in stems)}/{len(stems)}, "
          f"range [{min(stems):+.2f}, {max(stems):+.2f}], mean {sum(stems)/len(stems):+.2f}")
    print(f"Lambda(C_STEM) range: [{min(map(logistic, stems)):.2f}, {max(map(logistic, stems)):.2f}]")
    print(f"STEM V: positive in {sum(v > 0 for v in v_stems)}/{len(v_stems)}, "
          f"range [{min(v_stems):+.2f}, {max(v_stems):+.2f}], mean {sum(v_stems)/len(v_stems):+.2f}")
    print(f"|V| < |C| for STEM in "
          f"{sum(abs(a['STEM']) < abs(c['STEM']) for _, _, c, a in rows if a['STEM'] is not None)}"
          f"/{len(v_stems)} settings")
    for label in LABELS:
        vals = [a[label] for _, _, _, a in rows if a[label] is not None]
        print(f"  V[{label}]: n={len(vals)}, positive {sum(v > 0 for v in vals)}/{len(vals)}, "
              f"range [{min(vals):+.2f}, {max(vals):+.2f}], mean |V| {sum(map(abs, vals))/len(vals):.3f}")


if __name__ == "__main__":
    main()
