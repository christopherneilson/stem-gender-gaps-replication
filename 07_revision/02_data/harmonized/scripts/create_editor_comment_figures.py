#!/usr/bin/env python3
"""Create panel-level PDFs for editor response figures.

Outputs panel PDFs named figureX_panelY.pdf in harmonized/figures:
- Figure 1: panels a-c (top 5, top 10, top 20) for female share among outcome
  participants. Superseded in the manuscript by the stacked versions below, but
  still produced so the grouped-bar variant remains available.
- Stacked male/female shares: stacked_shares_top10.pdf is the main-text Figure 1
  and stacked_shares_top5/top20.pdf are the two Online Appendix B.I panels.
  figure1_stem_first_top10_stacked.pdf is the first-choice-only variant carried
  over from the original submission and is no longer included anywhere.
- Figure 2: panels a-f (pipeline composition/gap for top 5, top 10, top 20)
- Figures 3-5: panels a-f each for stem_first, stem_any, stem_admitted

Every panel is drawn from a CSV in ``harmonized/figure_data`` named for the
manuscript figure it feeds, written and read back by ``figure_data.py``. The
panel PDFs keep the names above because the .tex wrappers point at them; the
CSVs are named for the figure number a reader sees in the paper.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

import figure_data as fd

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "harmonized" / "master" / "harmonized_counts_long.csv"
OUT_DIR = BASE / "harmonized" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_VARIANT = {
    "Australia": "atar_pooled_2009_2010",
    "Brazil": "reply_tables",
    "Chile": "chile_2012_2018",
    "China": "china_ningxia_2018",
    "Finland": "main_pooled",
    "Greece": "years_2003_2012",
    "Spain": "esp_2020_2024_tables",
    "Sweden": "reply_docx_2017",
    "Taiwan": "reply_pdf_2010",
    "Uganda": "reply_docx",
}

COUNTRY_ORDER = [
    "Taiwan",
    "Chile",
    "Brazil",
    "China",
    "Uganda",
    "Finland",
    "Spain",
    "Australia",
    "Greece",
    "Sweden",
]

CHILE_ANALYSIS_CANDIDATE_FILES = [
    Path("confidential/chile_gender_analysis.dta"),
    BASE / "chile_gender_analysis.dta",
]
CHILE_DENOM_CACHE = BASE / "harmonized" / "cache" / "chile_applicant_denominators.csv"

THRESHOLDS = [
    ("Top 5%", [("top_percent", 5)]),
    ("Top 10%", [("decile", 10)]),
    ("Top 20%", [("decile", 9), ("decile", 10)]),
]


def _all_thresholds_to(filename: str, figure: str) -> dict[str, tuple[str, str]]:
    """Route all three thresholds of one outcome into a single CSV."""
    return {t: (filename, figure) for t in ("Top 5%", "Top 10%", "Top 20%")}


# Which manuscript exhibit each (panel family, threshold) pair feeds, as
# {threshold: (csv filename, figure number)}. Two thresholds mapping to one
# filename land in one CSV: Online Appendix B.II shows the top 5% and the top
# 20% in a single figure, so a coauthor looking it up should open a single file.
# The top-10% rows split off because the paper prints them as a main-text
# figure. Keep these in sync with figure_data/README.md.
PIPELINE_DEST = {
    "Top 10%": ("figure_02_pipeline_top10.csv", "Figure 2"),
    "Top 5%": ("figure_B02_pipeline_top5_top20.csv", "Figure B.II"),
    "Top 20%": ("figure_B02_pipeline_top5_top20.csv", "Figure B.II"),
}

OUTCOME_DEST = {
    "stem_first": {
        "Top 10%": ("figure_03_stem_first_top10.csv", "Figure 3"),
        "Top 5%": ("figure_B03_stem_first_top5_top20.csv", "Figure B.III"),
        "Top 20%": ("figure_B03_stem_first_top5_top20.csv", "Figure B.III"),
    },
    "stem_any": _all_thresholds_to(
        "figure_B04_stem_any_top5_top10_top20.csv", "Figure B.IV"
    ),
    "business_first": _all_thresholds_to(
        "figure_C01_business_first_top5_top10_top20.csv", "Figure C.I"
    ),
    "business_any": _all_thresholds_to(
        "figure_C02_business_any_top5_top10_top20.csv", "Figure C.II"
    ),
    "health_first": _all_thresholds_to(
        "figure_C03_health_first_top5_top10_top20.csv", "Figure C.III"
    ),
    "health_any": _all_thresholds_to(
        "figure_C04_health_any_top5_top10_top20.csv", "Figure C.IV"
    ),
    # Produced but not included by any manuscript figure.
    "stem_admitted": _all_thresholds_to(
        f"{fd.UNUSED_DIR}/stem_admitted_top5_top10_top20.csv", fd.UNUSED_LABEL
    ),
    "health_admitted": _all_thresholds_to(
        f"{fd.UNUSED_DIR}/health_admitted_top5_top10_top20.csv", fd.UNUSED_LABEL
    ),
    "business_admitted": _all_thresholds_to(
        f"{fd.UNUSED_DIR}/business_admitted_top5_top10_top20.csv", fd.UNUSED_LABEL
    ),
}

STACKED_DEST = {
    "Top 10%": ("figure_01_stem_applicant_shares_top10.csv", "Figure 1"),
    "Top 5%": ("figure_B01_stem_applicant_shares_top5_top20.csv", "Figure B.I"),
    "Top 20%": ("figure_B01_stem_applicant_shares_top5_top20.csv", "Figure B.I"),
}

OUTCOME_LABELS = {
    "stem_first": "STEM first",
    "stem_any": "Any STEM in list",
    "stem_admitted": "Admitted STEM",
    "health_first": "Health first",
    "health_any": "Any Health in list",
    "health_admitted": "Admitted Health",
    "business_first": "Business/Law first",
    "business_any": "Any Business/Law in list",
    "business_admitted": "Admitted Business/Law",
}

# Three ordered steps of the navy ramp, for panels whose bars are outcomes rather
# than sexes. Teal is deliberately not among them: it is reserved for "female"
# below, so no bar in the paper carries it while meaning something else.
# (CVD ΔE 17.3 worst adjacent, normal-vision 22.3.)
OUTCOME_COLORS = {
    "stem_first": "#162678",
    "stem_any": "#5627FF",
    "stem_admitted": "#7D92FE",
}

# Two far-apart steps of the brand ramp, one per sex. Deliberately two distinct
# hues rather than a dark and a light step of one hue: in a paper about gender
# gaps, rendering one sex as a lighter wash of the other encodes a hierarchy we
# do not intend. Colour-blind separation is deutan ΔE 53.3 (normal-vision 55.7),
# and the greyscale values are #30 against #D3 -- 163 levels apart, so the bars
# stay distinct in a black-and-white print. Teal sits at 1.46:1 against white,
# below the 3:1 contrast floor, which the direct value label on every bar and the
# hairline black edge relieve. Label colours are paired to each fill: white on
# the dark navy, black on the light teal.
MALE_COLOR = "#162678"
FEMALE_COLOR = "#38E9E9"
MALE_MARK_COLOR = "white"
FEMALE_MARK_COLOR = "black"

# The gap panels are coloured by which sex is ahead: positive means the female
# share is higher, negative the male share, so each bar takes the colour of the
# side it favors and the two panels of a figure read consistently.
POS_GAP_COLOR = FEMALE_COLOR
NEG_GAP_COLOR = MALE_COLOR

F1_AXIS_LABEL_FS = 22
F1_TICK_LABEL_FS = 17
F1_LEGEND_FS = 17
F1_VALUE_LABEL_FS = 13

# Sizes for the single-panel stacked Figure 1, which is rendered at full
# \textwidth rather than the 0.70\textwidth used by the stacked panel version.
F1S_AXIS_LABEL_FS = 21
F1S_TICK_LABEL_FS = 18
F1S_LEGEND_FS = 18
F1S_VALUE_LABEL_FS = 16

# Sizes for the paired stacked panels in Online Appendix B, which pack 20 bars
# into the same canvas and are rendered at 0.80\textwidth.
F1P_AXIS_LABEL_FS = 16
F1P_TICK_LABEL_FS = 19
F1P_LEGEND_FS = 16
F1P_VALUE_LABEL_FS = 15

# Axis label matches the tick labels; a larger axis title overpowered the panel
# once these figures were enlarged on the page.
FX_AXIS_LABEL_FS = 20
FX_TICK_LABEL_FS = 20
FX_LEGEND_FS = 20
FX_VALUE_LABEL_FS = 15

GAP_YMIN = -44
GAP_YMAX = 44

_CHILE_APPLICANT_COUNTS_CACHE = None


def _first_existing_path(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _group_key(groups: list[tuple[str, float]]) -> str | None:
    group_set = {(g_type, float(g_val)) for g_type, g_val in groups}
    if group_set == {("top_percent", 5.0)}:
        return "top5"
    if group_set == {("decile", 10.0)}:
        return "top10"
    if group_set == {("decile", 9.0), ("decile", 10.0)}:
        return "top20"
    return None


def _load_chile_cached_denominators() -> dict[str, dict[str, int]]:
    """Fall back to the committed snapshot of Chile's applicant-only denominators.

    The student-level file lives in a Dropbox folder only some coauthors have mounted.
    Returning {} here instead would silently drop the applicant restriction while the
    figure notes still claim it, so prefer the snapshot and say so.
    """
    if not CHILE_DENOM_CACHE.exists():
        print("  [chile] student-level file not found and no cached denominators; "
              "figures will use score-holder denominators, contradicting the figure notes")
        return {}
    c = pd.read_csv(CHILE_DENOM_CACHE, comment="#").set_index("score_group")
    dec = lambda i: c.loc[f"Decile {i}"]
    out = {
        "top5": {"male": int(c.loc["Top 5%", "male"]), "female": int(c.loc["Top 5%", "female"])},
        "top10": {"male": int(dec(10)["male"]), "female": int(dec(10)["female"])},
        "top20": {"male": int(dec(9)["male"] + dec(10)["male"]),
                  "female": int(dec(9)["female"] + dec(10)["female"])},
    }
    print(f"  [chile] student-level file not found; applicant denominators from {CHILE_DENOM_CACHE.name}")
    return out


def _load_chile_choice_applicant_denominators() -> dict[str, dict[str, int]]:
    global _CHILE_APPLICANT_COUNTS_CACHE
    if _CHILE_APPLICANT_COUNTS_CACHE is not None:
        return _CHILE_APPLICANT_COUNTS_CACHE

    path = _first_existing_path(CHILE_ANALYSIS_CANDIDATE_FILES)
    if path is None:
        _CHILE_APPLICANT_COUNTS_CACHE = _load_chile_cached_denominators()
        return _CHILE_APPLICANT_COUNTS_CACHE

    needed_cols = ["score_decile", "top_5", "female", "applies_university"]
    d = pd.read_stata(path, convert_categoricals=False, columns=needed_cols)
    d = d[d["applies_university"] == 1].copy()
    d = d[d["score_decile"].notna()].copy()
    d["score_decile"] = d["score_decile"].astype(int)

    out: dict[str, dict[str, int]] = {}
    masks = {
        "top5": d["top_5"] == 1,
        "top10": d["score_decile"] == 10,
        "top20": d["score_decile"].isin([9, 10]),
    }
    for key, mask in masks.items():
        tmp = d[mask]
        out[key] = {
            "male": int((tmp["female"] == 0).sum()),
            "female": int((tmp["female"] == 1).sum()),
        }

    _CHILE_APPLICANT_COUNTS_CACHE = out
    return _CHILE_APPLICANT_COUNTS_CACHE


def _select_preferred(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["country"].isin(PREFERRED_VARIANT.keys())].copy()
    d = d[d.apply(lambda r: r["sample_variant"] == PREFERRED_VARIANT.get(r["country"]), axis=1)]
    d["country"] = pd.Categorical(d["country"], categories=COUNTRY_ORDER, ordered=True)
    return d.sort_values(["country", "outcome", "score_group_type", "score_group_value", "sex"]).reset_index(drop=True)


def _subset_groups(df: pd.DataFrame, groups: list[tuple[str, float]]) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for g_type, g_val in groups:
        mask = mask | ((df["score_group_type"] == g_type) & (df["score_group_value"] == float(g_val)))
    return df[mask].copy()


def _pivot_counts(df: pd.DataFrame, outcome: str, groups: list[tuple[str, float]]) -> pd.DataFrame:
    d = _subset_groups(df[df["outcome"] == outcome], groups)
    if d.empty:
        return pd.DataFrame(columns=["country", "male", "female"])
    # observed=True: a country with no rows for this outcome (Greece reports no
    # business/law counts at the top 10% and 20%) must come out missing, not as
    # a zero count. pandas 3 does this by default; pandas 2 fills zeros.
    w = d.pivot_table(index=["country"], columns="sex", values="n", aggfunc="sum", observed=True).reset_index()
    w.columns.name = None
    for col in ["male", "female"]:
        if col not in w.columns:
            w[col] = np.nan
    w["country"] = pd.Categorical(w["country"], categories=COUNTRY_ORDER, ordered=True)
    return w.sort_values("country").reset_index(drop=True)[["country", "male", "female"]]


def _build_outcome_probs(df: pd.DataFrame, groups: list[tuple[str, float]], outcome: str) -> pd.DataFrame:
    pipe = _pivot_counts(df, "pipeline_total", groups).rename(columns={"male": "pipeline_male", "female": "pipeline_female"})
    out = _pivot_counts(df, outcome, groups).rename(columns={"male": "outcome_male", "female": "outcome_female"})
    merged = pipe.merge(out, on="country", how="inner")

    # For Chile STEM outcomes, condition probabilities on applicants only.
    if outcome in {"stem_first", "stem_any", "stem_admitted"}:
        g_key = _group_key(groups)
        chile_denom = _load_chile_choice_applicant_denominators()
        if g_key in chile_denom:
            cm = chile_denom[g_key].get("male")
            cf = chile_denom[g_key].get("female")
            if cm and cf:
                is_chile = merged["country"] == "Chile"
                merged.loc[is_chile, "pipeline_male"] = cm
                merged.loc[is_chile, "pipeline_female"] = cf

    # Stored in the units the paper prints -- percent for levels, percentage
    # points for gaps -- so a value read off a bar can be found in the CSV
    # without converting anything. Bar labels round to whole numbers; the CSV
    # keeps full precision.
    merged["prob_male_pct"] = merged["outcome_male"] / merged["pipeline_male"] * 100
    merged["prob_female_pct"] = merged["outcome_female"] / merged["pipeline_female"] * 100
    merged["gap_female_minus_male_pp"] = merged["prob_female_pct"] - merged["prob_male_pct"]

    # Some outcomes (e.g. business/health) have no rows for a few countries, so the
    # inner merge above can yield fewer than 10 rows. Reindex to the full country
    # order so the plotting functions always receive one row per country; missing
    # countries become NaN and simply render as absent bars.
    merged["country"] = merged["country"].astype(str)
    merged = merged.set_index("country").reindex(COUNTRY_ORDER).reset_index()
    return merged


def _build_pipeline_shares(df: pd.DataFrame, groups: list[tuple[str, float]]) -> pd.DataFrame:
    pipe = _pivot_counts(df, "pipeline_total", groups).rename(columns={"male": "n_male", "female": "n_female"})
    # Same reindex as _build_outcome_probs: the plotting helpers read the column
    # out positionally against COUNTRY_ORDER, so a country missing from the
    # master must leave a hole rather than shift every bar to its left.
    pipe["country"] = pipe["country"].astype(str)
    pipe = pipe.set_index("country").reindex(COUNTRY_ORDER).reset_index()
    pipe["n_total"] = pipe["n_male"] + pipe["n_female"]
    pipe["male_share_in_top_group_pct"] = pipe["n_male"] / pipe["n_total"] * 100
    pipe["female_share_in_top_group_pct"] = pipe["n_female"] / pipe["n_total"] * 100
    pipe["pipeline_gap_female_minus_male_pp"] = pipe["female_share_in_top_group_pct"] - pipe["male_share_in_top_group_pct"]
    return pipe


def _annotate_bars_percent(ax, bars, value_label_fs: int, dy: float = 1.0) -> None:
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            f"{h:.0f}%",
            ha="center",
            va="bottom",
            fontsize=value_label_fs,
        )


def _annotate_bars_percent_paired(
    ax,
    bars_by_series,
    value_label_fs: int,
    base_dy: float = 0.8,
    stagger: float = 4.5,
    overlap_thresh: float = 6.0,
) -> None:
    """Label grouped bars, raising the taller bar's label when the two bars in a
    group are close in height so their percent labels do not overlap.

    A label is wider than its own bar, so it always spills a little onto the
    neighbour. Where that neighbour is the taller, darker bar the black text
    would be unreadable, so each label carries a white backing box; over the
    plot background the box is invisible."""
    n_groups = len(bars_by_series[0])
    for gi in range(n_groups):
        group = [series[gi] for series in bars_by_series]
        heights = [b.get_height() for b in group]
        dys = [base_dy] * len(group)
        if (
            len(group) == 2
            and all(np.isfinite(h) for h in heights)
            and abs(heights[0] - heights[1]) < overlap_thresh
        ):
            taller = 0 if heights[0] >= heights[1] else 1
            dys[taller] = base_dy + stagger
        for b, dy in zip(group, dys):
            h = b.get_height()
            if not np.isfinite(h):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(h + dy, 98.5),
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=value_label_fs,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
            )


def _annotate_bars_percent_vertical(ax, bars_list, value_label_fs: int, dy: float = 1.0) -> None:
    """Place percent labels rotated vertically above each bar. Robust for dense
    grouped panels (e.g. business/health) where the values cluster in a narrow
    band, so horizontal labels of adjacent bars would collide side-to-side."""
    for bars in bars_list:
        for b in bars:
            h = b.get_height()
            if not np.isfinite(h):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + dy,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=value_label_fs,
            )


def _annotate_gap_bars(ax, bars, value_label_fs: int) -> None:
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        if h >= 0:
            y = min(h + 0.7, GAP_YMAX - 0.6)
            va = "bottom"
        else:
            y = max(h - 0.9, GAP_YMIN + 0.6)
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{h:.0f}%",
            ha="center",
            va=va,
            fontsize=value_label_fs,
        )


def _plot_gender_shares_among_outcomes(
    df: pd.DataFrame, threshold_label: str, out_path: Path, outcomes: list[str]
) -> None:
    x = np.arange(len(COUNTRY_ORDER))
    width = 0.32

    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    center = (len(outcomes) - 1) / 2
    bars_by_series = []
    for i, outcome in enumerate(outcomes):
        tmp = df[df["outcome"] == outcome].set_index("country").reindex(COUNTRY_ORDER)
        y = tmp["female_share_among_outcome_pct"].to_numpy()
        bars = ax.bar(
            x + (i - center) * width,
            y,
            width=width,
            label=OUTCOME_LABELS[outcome],
            color=OUTCOME_COLORS[outcome],
            edgecolor="black",
            linewidth=0.7,
        )
        bars_by_series.append(bars)
    _annotate_bars_percent_paired(ax, bars_by_series, value_label_fs=F1_VALUE_LABEL_FS)

    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(COUNTRY_ORDER, fontsize=F1_TICK_LABEL_FS, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=F1_TICK_LABEL_FS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Female share among\nstudents in outcome (%)", fontsize=F1_AXIS_LABEL_FS)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True, edgecolor="black", ncol=len(outcomes), fontsize=F1_LEGEND_FS)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.25)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_composition(df: pd.DataFrame, threshold_label: str, y_label: str, title_prefix: str, out_path: Path, value_labels: str = "paired") -> None:
    x = np.arange(len(COUNTRY_ORDER))
    width = 0.38
    male = df["prob_male_pct"].to_numpy()
    female = df["prob_female_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    bars_m = ax.bar(x - width / 2, male, width=width, color=MALE_COLOR, edgecolor="black", linewidth=0.7, label="Male")
    bars_f = ax.bar(x + width / 2, female, width=width, color=FEMALE_COLOR, edgecolor="black", linewidth=0.7, label="Female")
    if value_labels == "vertical":
        _annotate_bars_percent_vertical(ax, [bars_m, bars_f], value_label_fs=FX_VALUE_LABEL_FS)
    else:
        _annotate_bars_percent_paired(ax, [bars_m, bars_f], value_label_fs=FX_VALUE_LABEL_FS)
    ax.set_xticks(x)
    ax.set_xticklabels(COUNTRY_ORDER, fontsize=FX_TICK_LABEL_FS, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=FX_TICK_LABEL_FS)
    ax.set_ylim(0, 100)
    ax.set_ylabel(y_label, fontsize=FX_AXIS_LABEL_FS)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True, edgecolor="black", fontsize=FX_LEGEND_FS)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.25)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pipeline_composition(df: pd.DataFrame, threshold_label: str, out_path: Path) -> None:
    x = np.arange(len(COUNTRY_ORDER))
    width = 0.38
    male = df["male_share_in_top_group_pct"].to_numpy()
    female = df["female_share_in_top_group_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    bars_m = ax.bar(x - width / 2, male, width=width, color=MALE_COLOR, edgecolor="black", linewidth=0.7, label="Male")
    bars_f = ax.bar(x + width / 2, female, width=width, color=FEMALE_COLOR, edgecolor="black", linewidth=0.7, label="Female")
    _annotate_bars_percent_paired(ax, [bars_m, bars_f], value_label_fs=FX_VALUE_LABEL_FS)
    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(COUNTRY_ORDER, fontsize=FX_TICK_LABEL_FS, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=FX_TICK_LABEL_FS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share among\ntop score group (%)", fontsize=FX_AXIS_LABEL_FS)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True, edgecolor="black", fontsize=FX_LEGEND_FS)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.25)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_gap(df: pd.DataFrame, threshold_label: str, gap_col: str, title_prefix: str, out_path: Path) -> None:
    x = np.arange(len(COUNTRY_ORDER))
    gap_pp = df[gap_col].to_numpy()
    colors = [POS_GAP_COLOR if v >= 0 else NEG_GAP_COLOR for v in gap_pp]

    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    bars = ax.bar(x, gap_pp, color=colors, edgecolor="black", linewidth=0.7, width=0.58)
    _annotate_gap_bars(ax, bars, value_label_fs=FX_VALUE_LABEL_FS)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(COUNTRY_ORDER, fontsize=FX_TICK_LABEL_FS, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=FX_TICK_LABEL_FS)
    ax.set_ylim(GAP_YMIN, GAP_YMAX)
    ax.set_ylabel("Female minus\nmale (pp)", fontsize=FX_AXIS_LABEL_FS)
    ax.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.25)
    fig.savefig(out_path)
    plt.close(fig)


def _figure1_panels(df: pd.DataFrame) -> None:
    outcomes = ["stem_first", "stem_any"]
    built = {}
    for thr_label, groups in THRESHOLDS:
        rows = []
        for outcome in outcomes:
            num = _pivot_counts(df, outcome, groups).rename(columns={"male": "num_male", "female": "num_female"})
            num["outcome"] = outcome
            tot = num["num_male"] + num["num_female"]
            num["female_share_among_outcome_pct"] = num["num_female"] / tot * 100
            rows.append(num)
        panel_df = pd.concat(rows, ignore_index=True)
        panel_df.insert(0, "threshold", thr_label)
        built[thr_label] = panel_df

    published = fd.publish_grouped(
        built,
        _all_thresholds_to(
            f"{fd.UNUSED_DIR}/female_share_among_stem_applicants_grouped.csv", fd.UNUSED_LABEL
        ),
        key="threshold",
    )
    for panel, (thr_label, _) in zip(["a", "b", "c"], THRESHOLDS):
        _plot_gender_shares_among_outcomes(
            published[thr_label], thr_label, OUT_DIR / f"figure1_panel{panel}.pdf", outcomes
        )


def _figure1_stacked_top10(df: pd.DataFrame) -> None:
    """Single-panel stacked composition of top-10% STEM-first applicants by sex.

    This reproduces the headline figure of the original submission, but is built
    from the harmonized counts so the values agree with Figures 2-4 and with the
    top-5/10/20 panel version now in Online Appendix B.
    """
    thr_label, groups = THRESHOLDS[1]
    counts = _pivot_counts(df, "stem_first", groups)
    counts = counts.set_index("country").reindex(COUNTRY_ORDER).reset_index()
    total = counts["male"] + counts["female"]
    counts = counts.rename(columns={"male": "n_male", "female": "n_female"})
    counts.insert(0, "threshold", thr_label)
    counts["female_share_pct"] = counts["n_female"] / total * 100
    counts["male_share_pct"] = counts["n_male"] / total * 100
    counts = fd.publish(
        counts,
        filename=f"{fd.UNUSED_DIR}/stem_first_top10_stacked_single_panel.csv",
        figure=fd.UNUSED_LABEL,
    )
    share_female = counts["female_share_pct"].to_numpy()
    share_male = counts["male_share_pct"].to_numpy()

    x = np.arange(len(COUNTRY_ORDER))
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    bars_f = ax.bar(
        x, share_female, width=0.62, color=FEMALE_COLOR, edgecolor="black", linewidth=0.7, label="Female"
    )
    bars_m = ax.bar(
        x,
        share_male,
        width=0.62,
        bottom=share_female,
        color=MALE_COLOR,
        edgecolor="black",
        linewidth=0.7,
        label="Male",
    )

    for bar, value in zip(bars_f, share_female):
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value / 2,
            f"{value:.0f}%",
            ha="center",
            va="center",
            color=FEMALE_MARK_COLOR,
            fontsize=F1S_VALUE_LABEL_FS,
        )
    for bar, base, value in zip(bars_m, share_female, share_male):
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            base + value / 2,
            f"{value:.0f}%",
            ha="center",
            va="center",
            color=MALE_MARK_COLOR,
            fontsize=F1S_VALUE_LABEL_FS,
        )

    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    # Ten short country names fit horizontally at full \textwidth, as in the
    # original figure; the other panels rotate only because they are narrower.
    ax.set_xticklabels(COUNTRY_ORDER, fontsize=F1S_TICK_LABEL_FS)
    ax.tick_params(axis="y", labelsize=F1S_TICK_LABEL_FS)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("Gender shares among\nSTEM applicants (%)", fontsize=F1S_AXIS_LABEL_FS)
    ax.grid(axis="y", alpha=0.2)
    # Handles are reversed so the legend reads Male then Female, matching the
    # top-to-bottom order of the stacked segments.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        frameon=True,
        edgecolor="black",
        ncol=2,
        fontsize=F1S_LEGEND_FS,
        loc="lower right",
    )
    fig.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.11)
    fig.savefig(OUT_DIR / "figure1_stem_first_top10_stacked.pdf")
    plt.close(fig)


def _stacked_shares_by_threshold(df: pd.DataFrame) -> None:
    """Stacked male/female composition of STEM applicants, by threshold.

    Same logic as the main-text Figure 1, but with two stacked bars per country:
    the left bar covers students who rank a STEM program first and the right bar
    students who rank one anywhere on their list. Each bar runs the full 0-100
    range, so the male share is read off directly rather than inferred.

    Sex is carried by fill colour and the STEM definition by hatching, so the
    x axis only has to carry country names.

    Files are named by threshold rather than panel letter because the top-10
    version is the main-text Figure 2 while the other two are Online Appendix B
    panels, so a shared a/b/c numbering would be misleading.
    """
    outcomes = ["stem_first", "stem_any"]
    hatches = ["", "//"]
    # Bars sit 0.05 apart inside a pair and 0.21 apart between countries. The
    # solid/hatched contrast carries the pairing, so the bars can run wider than
    # spacing alone would allow.
    offsets = [-0.215, 0.215]
    width = 0.38
    group_x = np.arange(len(COUNTRY_ORDER), dtype=float)

    built = {}
    for thr_label, groups in THRESHOLDS:
        rows = []
        for outcome in outcomes:
            counts = _pivot_counts(df, outcome, groups)
            counts = counts.set_index("country").reindex(COUNTRY_ORDER).reset_index()
            total = counts["male"] + counts["female"]
            counts = counts.rename(columns={"male": "n_male", "female": "n_female"})
            counts.insert(1, "outcome", outcome)
            counts["female_share_among_applicants_pct"] = counts["n_female"] / total * 100
            counts["male_share_among_applicants_pct"] = counts["n_male"] / total * 100
            rows.append(counts)
        tidy = pd.concat(rows, ignore_index=True)
        tidy.insert(0, "threshold", thr_label)
        built[thr_label] = tidy

    published = fd.publish_grouped(built, STACKED_DEST, key="threshold")

    for slug, (thr_label, _) in zip(["top5", "top10", "top20"], THRESHOLDS):
        panel = published[thr_label]
        fig, ax = plt.subplots(figsize=(12.6, 5.9))
        for i, outcome in enumerate(outcomes):
            counts = panel[panel["outcome"] == outcome]
            share_female = counts["female_share_among_applicants_pct"].to_numpy()
            share_male = counts["male_share_among_applicants_pct"].to_numpy()

            pos = group_x + offsets[i]
            bars_f = ax.bar(
                pos,
                share_female,
                width=width,
                color=FEMALE_COLOR,
                edgecolor="black",
                linewidth=0.7,
            )
            bars_m = ax.bar(
                pos,
                share_male,
                width=width,
                bottom=share_female,
                color=MALE_COLOR,
                edgecolor="black",
                linewidth=0.7,
            )
            if hatches[i]:
                # matplotlib 3.10 ties hatch colour to edgecolor, so a black
                # hatch would vanish on the dark green. Overlay an unfilled,
                # unstroked bar whose edgecolor sets only the hatch.
                for base, values, mark in (
                    (None, share_female, FEMALE_MARK_COLOR),
                    (share_female, share_male, MALE_MARK_COLOR),
                ):
                    ax.bar(
                        pos,
                        values,
                        width=width,
                        bottom=base,
                        facecolor="none",
                        edgecolor=mark,
                        linewidth=0.0,
                        hatch=hatches[i],
                    )
            # Labels are rotated because 20 bars leave each one narrower than a
            # horizontal "100%", which would spill over the bar edges. Each sits
            # on a patch of its own segment colour so the hatching on the
            # any-STEM bars does not run through the digits.
            for bar, value in zip(bars_f, share_female):
                if not np.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    rotation=90,
                    color=FEMALE_MARK_COLOR,
                    fontsize=F1P_VALUE_LABEL_FS,
                    bbox=dict(facecolor=FEMALE_COLOR, edgecolor="none", pad=1.2),
                )
            for bar, base, value in zip(bars_m, share_female, share_male):
                if not np.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    rotation=90,
                    color=MALE_MARK_COLOR,
                    fontsize=F1P_VALUE_LABEL_FS,
                    bbox=dict(facecolor=MALE_COLOR, edgecolor="none", pad=1.2),
                )

        ax.axhline(50, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.tick_params(axis="y", labelsize=F1P_TICK_LABEL_FS)
        ax.set_ylabel("Gender shares among\nSTEM applicants (%)", fontsize=F1P_AXIS_LABEL_FS)
        ax.grid(axis="y", alpha=0.2)

        ax.set_xticks(group_x)
        ax.set_xticklabels(COUNTRY_ORDER, fontsize=F1P_TICK_LABEL_FS, rotation=45, ha="right")
        ax.set_xlim(-0.62, len(COUNTRY_ORDER) - 0.38)

        # Colour carries sex, hatching carries the STEM definition, so the legend
        # is built by hand from one swatch per dimension rather than from the
        # bars themselves. Every bar reaches 100, so it sits outside the axes,
        # below the rotated country names.
        legend_handles = [
            Patch(facecolor=MALE_COLOR, edgecolor="black", linewidth=0.7, label="Male"),
            Patch(facecolor=FEMALE_COLOR, edgecolor="black", linewidth=0.7, label="Female"),
            Patch(facecolor="white", edgecolor="black", linewidth=0.7, label="STEM first"),
            Patch(facecolor="white", edgecolor="black", linewidth=0.7, hatch="///", label="Any STEM"),
        ]
        ax.legend(
            handles=legend_handles,
            frameon=True,
            edgecolor="black",
            ncol=4,
            fontsize=F1P_LEGEND_FS,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.275),
        )
        fig.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=0.295)
        fig.savefig(OUT_DIR / f"stacked_shares_{slug}.pdf")
        plt.close(fig)


# Panel letters run (composition, gap) per threshold, in THRESHOLDS order.
PANEL_MAPPING = [
    ("a", "b", THRESHOLDS[0]),
    ("c", "d", THRESHOLDS[1]),
    ("e", "f", THRESHOLDS[2]),
]


def _figure2_panels(df: pd.DataFrame) -> None:
    built = {}
    for _, _, (thr_label, groups) in PANEL_MAPPING:
        pipe = _build_pipeline_shares(df, groups)
        pipe.insert(0, "threshold", thr_label)
        built[thr_label] = pipe

    published = fd.publish_grouped(built, PIPELINE_DEST, key="threshold")

    for comp_panel, gap_panel, (thr_label, _) in PANEL_MAPPING:
        pipe = published[thr_label]
        _plot_pipeline_composition(pipe, thr_label, OUT_DIR / f"figure2_panel{comp_panel}.pdf")
        _plot_gap(
            pipe,
            thr_label,
            "pipeline_gap_female_minus_male_pp",
            "Pipeline Gap",
            OUT_DIR / f"figure2_panel{gap_panel}.pdf",
        )


def _figure_outcome_panels(df: pd.DataFrame, figure_num: int, outcome: str, value_labels: str = "paired") -> None:
    built = {}
    for _, _, (thr_label, groups) in PANEL_MAPPING:
        wide = _build_outcome_probs(df, groups, outcome)
        wide.insert(0, "threshold", thr_label)
        wide.insert(1, "outcome", outcome)
        built[thr_label] = wide

    published = fd.publish_grouped(built, OUTCOME_DEST[outcome], key="threshold")

    for comp_panel, gap_panel, (thr_label, _) in PANEL_MAPPING:
        wide = published[thr_label]
        _plot_composition(
            wide,
            thr_label,
            # Two lines: rotated, a single line runs longer than the axes are
            # tall and the trailing "(%)" gets clipped.
            "Probability within\ntop score group (%)",
            OUTCOME_LABELS[outcome],
            OUT_DIR / f"figure{figure_num}_panel{comp_panel}.pdf",
            value_labels=value_labels,
        )
        _plot_gap(
            wide,
            thr_label,
            "gap_female_minus_male_pp",
            f"{OUTCOME_LABELS[outcome]} Gap",
            OUT_DIR / f"figure{figure_num}_panel{gap_panel}.pdf",
        )


def main() -> None:
    df = pd.read_csv(INPUT)
    df = _select_preferred(df)

    _figure1_panels(df)
    _figure1_stacked_top10(df)
    _stacked_shares_by_threshold(df)
    _figure2_panels(df)
    _figure_outcome_panels(df, 3, "stem_first")
    _figure_outcome_panels(df, 4, "stem_any")
    _figure_outcome_panels(df, 5, "stem_admitted")
    
    _figure_outcome_panels(df, 6, "health_first")
    _figure_outcome_panels(df, 7, "health_any")
    _figure_outcome_panels(df, 8, "health_admitted")
    
    _figure_outcome_panels(df, 9, "business_first")
    _figure_outcome_panels(df, 10, "business_any")
    _figure_outcome_panels(df, 11, "business_admitted")

    fd.write_column_dictionary()
    print(f"Wrote panel PDFs to {OUT_DIR}")


if __name__ == "__main__":
    main()
