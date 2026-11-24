#!/usr/bin/env python3
"""Emit one tidy CSV per plotted panel, straight from the figure code paths.

Every number here is produced by importing the scripts that draw the figures and
calling their own data-prep functions. Nothing is transcribed from the compiled
paper, and nothing is recomputed by hand -- doing so silently diverges, because
Chile's probability denominators are restricted to students with at least one
submitted application and Australia ships four sample variants of which only
`atar_pooled_2009_2010` is published.

Source modules (all under 07_revision/02_data/harmonized/scripts):
  create_editor_comment_figures.py            Figures 1-3, B.I-B.IV, C.I-C.IV
  create_online_appendix_distribution_figures.py   Figures B.V-B.VII
  create_parity_correlation_draft_figures.py  Figure 4
  create_appendix_d_recent_figures.py         Figure D.I

Schema: x, y, ci_lo, ci_hi, series, n_obs (+ context columns).

ci_lo/ci_hi are emitted and left empty on purpose. No figure in this paper plots
uncertainty: the bars are population counts over the full universe of applicants
in each admissions system, not sample estimates, and no script computes a
standard error. The columns are kept so their emptiness is explicit and the
validator can report it, rather than the absence passing unnoticed.

Usage:
  python figures/build_figure_data.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "07_revision" / "02_data" / "harmonized" / "scripts"
OUT = Path(__file__).resolve().parent / "data"

sys.path.insert(0, str(SCRIPTS))

import create_editor_comment_figures as EF          # noqa: E402
import create_online_appendix_distribution_figures as OA  # noqa: E402
import create_appendix_d_recent_figures as AD       # noqa: E402
import create_parity_correlation_draft_figures as PC  # noqa: E402

COLS = ["x", "series", "y", "ci_lo", "ci_hi", "n_obs"]

# THRESHOLDS index -> (levels panel file, gap panel file) in the generator's names
THRESH_IDX = {"Top 5%": 0, "Top 10%": 1, "Top 20%": 2}


def _write(df: pd.DataFrame, name: str, extra: list[str] | None = None) -> tuple[str, int]:
    cols = COLS + (extra or [])
    df = df[cols].copy()
    for c in ("y", "n_obs"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["y"] = df["y"].round(4)
    # Force LF. pandas defaults to os.linesep, so a Windows run would write CRLF,
    # git would normalize the blob to LF, and every SHA256 in checksums.sha256
    # would then mismatch on a macOS or Linux checkout. figures/data/*.csv is
    # pinned to eol=lf in .gitattributes to keep both ends of this consistent.
    df.to_csv(OUT / name, index=False, lineterminator="\n")
    return name, len(df)


# --------------------------------------------------------------------------
# Figures 1 and B.I -- stacked sex composition among STEM applicants
# --------------------------------------------------------------------------

def composition_panel(df, threshold: str) -> pd.DataFrame:
    _, groups = EF.THRESHOLDS[THRESH_IDX[threshold]]
    rows = []
    for outcome in ["stem_first", "stem_any"]:
        counts = EF._pivot_counts(df, outcome, groups)
        counts = counts.set_index("country").reindex(EF.COUNTRY_ORDER).reset_index()
        for _, r in counts.iterrows():
            total = (r["male"] or 0) + (r["female"] or 0)
            for sex in ["female", "male"]:
                rows.append({
                    "x": r["country"],
                    "series": f"{sex} / {EF.OUTCOME_LABELS.get(outcome, outcome)}",
                    "y": 100 * r[sex] / total if total else None,
                    "ci_lo": None, "ci_hi": None,
                    "n_obs": total,
                    "sex": sex, "outcome": outcome, "threshold": threshold,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main-text Figures 1, 3 and 4 as gallery.py draws them
# --------------------------------------------------------------------------
# gallery.py reads these three from fig1_revised.csv, fig6.csv and figR0.csv.
# The files were first written by one-off code during the August 2026 figure
# redesign; these functions rebuild them from the master, byte for byte.

ALL_DECILES = [("decile", i) for i in range(1, 11)]


def _top10_counts(df, outcome: str, groups=None) -> pd.DataFrame:
    _, top10 = EF.THRESHOLDS[THRESH_IDX["Top 10%"]]
    return EF._pivot_counts(df, outcome, groups or top10).set_index("country")


def _female_share(counts: pd.DataFrame, country: str) -> float:
    m, f = counts.loc[country, "male"], counts.loc[country, "female"]
    return 100 * f / (m + f)


def figure1_panel(df) -> pd.DataFrame:
    """Women's share of first-choice STEM applicants: all applicants vs top 10%."""
    groups = [
        ("all applicants", ALL_DECILES, "All deciles", "summed over deciles 1-10"),
        ("top 10% of scorers", None, "Decile 10", "in decile 10"),
    ]
    tables = {label: _top10_counts(df, "stem_first", g) for label, g, _, _ in groups}
    rows = []
    for country in EF.COUNTRY_ORDER:
        for label, _, score_group, where in groups:
            t = tables[label]
            rows.append({
                "x": country, "series": label,
                "y": _female_share(t, country),
                "ci_lo": None, "ci_hi": None,
                "n_obs": int(t.loc[country, "male"] + t.loc[country, "female"]),
                "score_group": score_group,
                "derivation": ("female stem_first count / total stem_first count "
                               f"{where} of the preferred sample variant"),
            })
    return pd.DataFrame(rows)


def figure3_panel(df) -> pd.DataFrame:
    """Pipeline component P = ln(T_m/T_f) on x, choice component
    C = ln(s_m/s_f) on y, top 10%, all score holders as denominators."""
    T, F = _top10_counts(df, "pipeline_total"), _top10_counts(df, "stem_first")
    rows = []
    for country in sorted(T.index.astype(str)):
        tm, tf = T.loc[country, "male"], T.loc[country, "female"]
        fm, ff = F.loc[country, "male"], F.loc[country, "female"]
        c = round(math.log((fm / tm) / (ff / tf)), 4)
        rows.append({
            "x": round(math.log(tm / tf), 4), "series": country, "y": c,
            "ci_lo": None, "ci_hi": None, "n_obs": None,
            "component_c": c, "male_share": round(fm / (fm + ff), 4),
        })
    return pd.DataFrame(rows)


def figure4_panel(df) -> pd.DataFrame:
    """Women's share at each stage, top 10%, settings ordered by the last stage."""
    stages = [
        ("In top 10%", _top10_counts(df, "pipeline_total")),
        ("Ranked STEM anywhere", _top10_counts(df, "stem_any")),
        ("Ranked STEM first", _top10_counts(df, "stem_first")),
        ("Assigned to STEM", _top10_counts(df, "stem_admitted")),
    ]
    assigned = stages[-1][1]
    order = sorted(assigned.index.astype(str), key=lambda c: _female_share(assigned, c))
    rows = []
    for country in order:
        for i, (label, t) in enumerate(stages):
            rows.append({
                "x": country, "series": label,
                "y": _female_share(t, country),
                "ci_lo": None, "ci_hi": None,
                "n_obs": int(t.loc[country, "male"] + t.loc[country, "female"]),
                "stage_order": i,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures 2, B.II -- pipeline composition and pipeline gap
# --------------------------------------------------------------------------

def pipeline_panels(df, threshold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, groups = EF.THRESHOLDS[THRESH_IDX[threshold]]
    pipe = EF._build_pipeline_shares(df, groups)
    lv, gp = [], []
    for _, r in pipe.iterrows():
        # Shares are taken from the counts rather than the helper's *_pct
        # columns: 100 * (n / total) is what the committed panels were written
        # with, and n / total * 100 can differ from it in the last bit.
        share = {sex: r[f"n_{sex}"] / r["n_total"] for sex in ["male", "female"]}
        for sex in ["male", "female"]:
            lv.append({
                "x": r["country"], "series": sex,
                "y": 100 * share[sex],
                "ci_lo": None, "ci_hi": None,
                "n_obs": r["n_total"], "threshold": threshold,
            })
        gp.append({
            "x": r["country"], "series": "female minus male",
            "y": 100 * (share["female"] - share["male"]),
            "ci_lo": None, "ci_hi": None,
            "n_obs": r["n_total"], "threshold": threshold,
        })
    return pd.DataFrame(lv), pd.DataFrame(gp)


# --------------------------------------------------------------------------
# Figures 3, 4-any, B.III, B.IV, C.I-C.IV -- outcome probability and gap
# --------------------------------------------------------------------------

def outcome_panels(df, outcome: str, threshold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, groups = EF.THRESHOLDS[THRESH_IDX[threshold]]
    wide = EF._build_outcome_probs(df, groups, outcome)
    lv, gp = [], []
    for _, r in wide.iterrows():
        note = "applicants only" if r["country"] == "Chile" else "all in score group"
        # As in pipeline_panels, rates come from the counts, not the *_pct columns.
        prob = {sex: r[f"outcome_{sex}"] / r[f"pipeline_{sex}"] for sex in ["male", "female"]}
        for sex in ["male", "female"]:
            lv.append({
                "x": r["country"], "series": sex,
                "y": 100 * prob[sex],
                "ci_lo": None, "ci_hi": None,
                "n_obs": r[f"pipeline_{sex}"],
                "n_outcome": r[f"outcome_{sex}"],
                "threshold": threshold, "outcome": outcome,
                "denominator": note,
            })
        denom = (r["pipeline_male"] or 0) + (r["pipeline_female"] or 0)
        gp.append({
            "x": r["country"], "series": "female minus male",
            "y": 100 * (prob["female"] - prob["male"]),
            "ci_lo": None, "ci_hi": None,
            "n_obs": denom,
            "threshold": threshold, "outcome": outcome,
            "denominator": note,
        })
    return pd.DataFrame(lv), pd.DataFrame(gp)


# --------------------------------------------------------------------------
# Figures B.V-B.VII -- score bins (bottom 50% pooled, then deciles 6-10)
# --------------------------------------------------------------------------

def _bin_denominators(df_oa, country: str, outcome: str | None) -> pd.Series:
    """Bin denominators, rebuilt with the same helpers _gap_by_decile uses.

    _female_share_by_decile and _gap_by_decile both return only (group, value),
    so the headcount behind each bar is not recoverable from their output. This
    repeats their first two steps -- including the Chile applicant-denominator
    swap -- and stops before the ratio, so n_obs comes from the same code path
    rather than a parallel reimplementation.
    """
    pipe = OA._counts_by_decile(df_oa, country, "pipeline_total").rename(
        columns={"male": "pipe_male", "female": "pipe_female"}
    )
    if country == "Chile" and outcome in {"stem_first", "stem_any", "stem_admitted"}:
        chile_app = OA._load_chile_decile_applicant_denominators()
        if not chile_app.empty:
            pipe = pipe[["decile"]].merge(chile_app, on="decile", how="left")
    agg = OA._aggregate_bottom_half_keep_top_deciles(pipe, "pipe_male", "pipe_female")
    return (agg["pipe_male"] + agg["pipe_female"]).reset_index(drop=True)


def bin_panel(df_oa, kind: str) -> pd.DataFrame:
    rows = []
    for country in OA.COUNTRY_ORDER_10:
        data_country = OA.DATA_COUNTRY_MAP.get(country, country)
        try:
            if kind == "female_share":
                d = OA._female_share_by_decile(df_oa, data_country)
                ycol = "female_share_pct"
                denom = _bin_denominators(df_oa, data_country, None)
            else:
                d = OA._gap_by_decile(df_oa, data_country, kind)
                ycol = "gap_female_minus_male_pp"
                denom = _bin_denominators(df_oa, data_country, kind)
        except Exception:
            continue
        if d is None or d.empty:
            continue
        d = d.reset_index(drop=True)
        for i, r in d.iterrows():
            rows.append({
                "x": r["score_bin"], "series": country,
                "y": r[ycol],
                "ci_lo": None, "ci_hi": None,
                "n_obs": denom.iloc[i] if i < len(denom) else None,
                "denominator": ("applicants only"
                                if country == "Chile" and kind != "female_share"
                                else "all in score bin"),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure 4 -- wage parity scatter (two panels)
# --------------------------------------------------------------------------

def parity_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Columns fixed to match _make_figure5_wage_only(), which draws the published
    # figure: both panels put wage_parity_index on x. Matching the column by
    # substring instead picks up wage_gap_percent and silently plots 1 - x.
    d = PC._load_inputs()
    xcol = "wage_parity_index"
    out = []
    for gapcol in ["pipeline_gap_female_minus_male_pp", "stem_first_gap_female_minus_male_pp"]:
        rows = []
        for _, r in d.iterrows():
            if pd.isna(r.get(xcol)) or pd.isna(r.get(gapcol)):
                continue
            rows.append({
                "x": r[xcol], "series": r["country"],
                "y": r[gapcol],
                "ci_lo": None, "ci_hi": None,
                "n_obs": None,
                "country": r["country"],
            })
        out.append(pd.DataFrame(rows))
    return out[0], out[1]


# --------------------------------------------------------------------------
# Figure D.I -- most recent cohort, top 10%
# --------------------------------------------------------------------------

def appendix_d_panel() -> pd.DataFrame:
    d = pd.read_csv(AD.INPUT)
    d = AD.top10_subset(AD.select_rows(d))
    g = AD.compute_gaps(d)
    rows = []
    labels = {
        "gap_pipeline_pp": "pipeline gap",
        "gap_stem_first_pp": "STEM first-choice gap",
        "gap_stem_any_pp": "any-STEM gap",
    }
    for _, r in g.iterrows():
        denom = (r.get("pipe_f") or 0) + (r.get("pipe_m") or 0)
        for col, lab in labels.items():
            rows.append({
                "x": r["country"], "series": lab,
                "y": r[col],
                "ci_lo": None, "ci_hi": None,
                "n_obs": denom,
                "years_used": r.get("years_used"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int]] = []

    df = EF._select_preferred(pd.read_csv(EF.INPUT))

    # Main-text Figures 1, 3 and 4 as gallery.py draws them
    written.append(_write(figure1_panel(df), "fig1_revised.csv", ["score_group", "derivation"]))
    written.append(_write(figure3_panel(df), "fig6.csv", ["component_c", "male_share"]))
    written.append(_write(figure4_panel(df), "figR0.csv", ["stage_order"]))

    # Figure 1 (main, top 10%) and B.I (top 5% / top 20%)
    written.append(_write(composition_panel(df, "Top 10%"), "fig1.csv",
                          ["sex", "outcome", "threshold"]))
    written.append(_write(composition_panel(df, "Top 5%"), "figB1_a.csv",
                          ["sex", "outcome", "threshold"]))
    written.append(_write(composition_panel(df, "Top 20%"), "figB1_b.csv",
                          ["sex", "outcome", "threshold"]))

    # Figure 2 (main, top 10%) and B.II (top 5% / top 20%)
    lv, gp = pipeline_panels(df, "Top 10%")
    written.append(_write(lv, "fig2_a.csv", ["threshold"]))
    written.append(_write(gp, "fig2_b.csv", ["threshold"]))
    for slug, thr in [("a", "Top 5%"), ("c", "Top 20%")]:
        lv, gp = pipeline_panels(df, thr)
        written.append(_write(lv, f"figB2_{slug}.csv", ["threshold"]))
        written.append(_write(gp, f"figB2_{chr(ord(slug) + 1)}.csv", ["threshold"]))

    # Figure 3 (main, stem_first top 10%) and B.III (top 5% / top 20%)
    lv, gp = outcome_panels(df, "stem_first", "Top 10%")
    written.append(_write(lv, "fig3_a.csv", ["threshold", "outcome", "denominator", "n_outcome"]))
    written.append(_write(gp, "fig3_b.csv", ["threshold", "outcome", "denominator"]))
    for slug, thr in [("a", "Top 5%"), ("c", "Top 20%")]:
        lv, gp = outcome_panels(df, "stem_first", thr)
        written.append(_write(lv, f"figB3_{slug}.csv",
                              ["threshold", "outcome", "denominator", "n_outcome"]))
        written.append(_write(gp, f"figB3_{chr(ord(slug) + 1)}.csv",
                              ["threshold", "outcome", "denominator"]))

    # B.IV -- stem_any at all three thresholds
    for i, thr in enumerate(["Top 5%", "Top 10%", "Top 20%"]):
        lv, gp = outcome_panels(df, "stem_any", thr)
        written.append(_write(lv, f"figB4_{chr(ord('a') + 2 * i)}.csv",
                              ["threshold", "outcome", "denominator", "n_outcome"]))
        written.append(_write(gp, f"figB4_{chr(ord('a') + 2 * i + 1)}.csv",
                              ["threshold", "outcome", "denominator"]))

    # Appendix C -- business and health, all three thresholds
    c_map = [("C1", "business_first"), ("C2", "business_any"),
             ("C3", "health_first"), ("C4", "health_any")]
    for fig_id, outcome in c_map:
        for i, thr in enumerate(["Top 5%", "Top 10%", "Top 20%"]):
            lv, gp = outcome_panels(df, outcome, thr)
            written.append(_write(lv, f"fig{fig_id}_{chr(ord('a') + 2 * i)}.csv",
                                  ["threshold", "outcome", "denominator", "n_outcome"]))
            written.append(_write(gp, f"fig{fig_id}_{chr(ord('a') + 2 * i + 1)}.csv",
                                  ["threshold", "outcome", "denominator"]))

    # B.V-B.VII -- score bins
    df_oa = OA._select_preferred(pd.read_csv(OA.INPUT))
    written.append(_write(bin_panel(df_oa, "female_share"), "figB5.csv", ["denominator"]))
    written.append(_write(bin_panel(df_oa, "stem_first"), "figB6.csv", ["denominator"]))
    written.append(_write(bin_panel(df_oa, "stem_any"), "figB7.csv", ["denominator"]))

    # Figure 4 -- wage parity scatter
    p_a, p_b = parity_panels()
    written.append(_write(p_a, "fig4_a.csv", ["country"]))
    written.append(_write(p_b, "fig4_b.csv", ["country"]))

    # Figure D.I
    written.append(_write(appendix_d_panel(), "figD1.csv", ["years_used"]))

    print(f"Wrote {len(written)} panel CSVs to {OUT}")
    for name, n in written:
        print(f"  {name:<18} {n:>4} rows")


if __name__ == "__main__":
    main()
