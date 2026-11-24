#!/usr/bin/env python3
"""Create Online Appendix figures across score bins.

Outputs 4 PDF figures (2x5 panels each):
- Figure A: female shares by score bin
- Figure B: STEM first-choice gender gap by score bin
- Figure C: Any-STEM gender gap by score bin
- Figure D: STEM admission gender gap by score bin

Score bins pool deciles 1-5 into one bar and keep deciles 6-10 separate.

If a country is unavailable in the harmonized master file, its panel is kept as
an explicit placeholder so the 10-panel structure is preserved.

Each figure is drawn from a CSV in ``harmonized/figure_data`` named for the
manuscript figure it feeds, written and read back by ``figure_data.py``.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

CHILE_ANALYSIS_CANDIDATE_FILES = [
    Path("confidential/chile_gender_analysis.dta"),
    BASE / "chile_gender_analysis.dta",
]
CHILE_DENOM_CACHE = BASE / "harmonized" / "cache" / "chile_applicant_denominators.csv"

COUNTRY_ORDER_10 = [
    "Taiwan",
    "Chile",
    "Brazil",
    "China (Ningxia)",
    "Uganda",
    "Finland",
    "Spain",
    "Australia",
    "Greece",
    "Sweden",
]

DATA_COUNTRY_MAP = {
    "China (Ningxia)": "China",
}

DECILES = list(range(1, 11))
GROUP_LABELS = ["1-5", "6", "7", "8", "9", "10"]
GROUP_POS = list(range(1, len(GROUP_LABELS) + 1))

# Manuscript exhibit each figure feeds, as (csv filename, figure number). The
# admitted-outcome figure is still produced but no longer included anywhere, so
# it is published to the not-in-manuscript subfolder. Keep in sync with
# figure_data/README.md.
FIGURE_DEST = {
    "female_share": ("figure_B05_female_shares_by_score_bin.csv", "Figure B.V"),
    "stem_first": ("figure_B06_stem_first_gap_by_score_bin.csv", "Figure B.VI"),
    "stem_any": ("figure_B07_stem_any_gap_by_score_bin.csv", "Figure B.VII"),
    "stem_admitted": (
        f"{fd.UNUSED_DIR}/stem_admitted_gap_by_score_bin.csv",
        fd.UNUSED_LABEL,
    ),
}

# Figure style. Same ramp as the main-text figures, and the gap panels follow the
# same rule there: a bar takes the colour of the sex it favors, so positive
# (female share higher) is the warm step and negative the dark purple. The share
# panels carry one series, so they take the same warm step rather than a fourth
# colour of their own.
BAR_COLOR_SHARE = "#38E9E9"
BAR_COLOR_POS = "#38E9E9"
BAR_COLOR_NEG = "#162678"
AX_LABEL_FS = 13
TICK_FS = 10
TITLE_FS = 11
VAL_FS = 8

_CHILE_DECILE_APPLICANT_CACHE = None


def _first_existing_path(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_chile_decile_applicant_denominators() -> pd.DataFrame:
    global _CHILE_DECILE_APPLICANT_CACHE
    if _CHILE_DECILE_APPLICANT_CACHE is not None:
        return _CHILE_DECILE_APPLICANT_CACHE

    path = _first_existing_path(CHILE_ANALYSIS_CANDIDATE_FILES)
    if path is None:
        # The student-level file lives in a Dropbox folder only some coauthors have mounted.
        # Returning an empty frame silently drops the applicant restriction while the figure
        # notes still claim it, so prefer the committed snapshot and say so.
        if CHILE_DENOM_CACHE.exists():
            c = pd.read_csv(CHILE_DENOM_CACHE, comment="#")
            c = c[c["score_group"].str.startswith("Decile")].copy()
            c["decile"] = c["score_group"].str.split().str[1].astype(int)
            _CHILE_DECILE_APPLICANT_CACHE = (
                c.rename(columns={"male": "pipe_male", "female": "pipe_female"})
                [["decile", "pipe_male", "pipe_female"]].sort_values("decile").reset_index(drop=True)
            )
            print(f"  [chile] student-level file not found; applicant denominators from {CHILE_DENOM_CACHE.name}")
        else:
            print("  [chile] student-level file not found and no cached denominators; "
                  "figures will use score-holder denominators, contradicting the figure notes")
            _CHILE_DECILE_APPLICANT_CACHE = pd.DataFrame(columns=["decile", "pipe_male", "pipe_female"])
        return _CHILE_DECILE_APPLICANT_CACHE

    d = pd.read_stata(
        path,
        convert_categoricals=False,
        columns=["score_decile", "female", "applies_university"],
    )
    d = d[(d["applies_university"] == 1) & (d["score_decile"].notna())].copy()
    d["decile"] = d["score_decile"].astype(int)
    grouped = d.groupby(["decile", "female"]).size().reset_index(name="n")
    wide = grouped.pivot_table(index="decile", columns="female", values="n", aggfunc="sum").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={0: "pipe_male", 1: "pipe_female"})
    for c in ["pipe_male", "pipe_female"]:
        if c not in wide.columns:
            wide[c] = np.nan

    _CHILE_DECILE_APPLICANT_CACHE = wide[["decile", "pipe_male", "pipe_female"]]
    return _CHILE_DECILE_APPLICANT_CACHE


def _select_preferred(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["country"].isin(PREFERRED_VARIANT.keys())].copy()
    d = d[d.apply(lambda r: r["sample_variant"] == PREFERRED_VARIANT.get(r["country"]), axis=1)]
    return d


def _counts_by_decile(df: pd.DataFrame, country: str, outcome: str) -> pd.DataFrame:
    d = df[
        (df["country"] == country)
        & (df["score_group_type"] == "decile")
        & (df["score_group_value"].isin([float(x) for x in DECILES]))
        & (df["outcome"] == outcome)
    ].copy()
    if d.empty:
        return pd.DataFrame(columns=["decile", "male", "female"])

    d["decile"] = d["score_group_value"].astype(int)
    wide = d.pivot_table(index="decile", columns="sex", values="n", aggfunc="sum").reset_index()
    wide.columns.name = None
    for col in ["male", "female"]:
        if col not in wide.columns:
            wide[col] = np.nan

    base = pd.DataFrame({"decile": DECILES})
    return base.merge(wide[["decile", "male", "female"]], on="decile", how="left")


def _aggregate_bottom_half_keep_top_deciles(
    t: pd.DataFrame,
    male_col: str,
    female_col: str,
) -> pd.DataFrame:
    """Return bins: 1-5 pooled, then deciles 6..10 separately."""
    out = []

    btm = t[t["decile"].between(1, 5)]
    out.append(
        {
            "group": "1-5",
            male_col: btm[male_col].sum(skipna=True),
            female_col: btm[female_col].sum(skipna=True),
        }
    )

    for d in range(6, 11):
        s = t[t["decile"] == d]
        out.append(
            {
                "group": str(d),
                male_col: s[male_col].sum(skipna=True),
                female_col: s[female_col].sum(skipna=True),
            }
        )

    out = pd.DataFrame(out)
    out["group"] = pd.Categorical(out["group"], categories=GROUP_LABELS, ordered=True)
    return out.sort_values("group").reset_index(drop=True)


def _female_share_by_decile(df: pd.DataFrame, country: str) -> pd.DataFrame:
    pipe = _counts_by_decile(df, country, "pipeline_total")
    agg = _aggregate_bottom_half_keep_top_deciles(pipe, "male", "female")
    agg["n_total"] = agg["male"] + agg["female"]
    agg["female_share_pct"] = (agg["female"] / agg["n_total"]) * 100
    # Counts ride along with the share so the published CSV can be checked
    # without going back to the master file.
    return agg.rename(
        columns={"group": "score_bin", "male": "n_male", "female": "n_female"}
    )[["score_bin", "n_male", "n_female", "n_total", "female_share_pct"]]


def _gap_by_decile(df: pd.DataFrame, country: str, outcome: str) -> pd.DataFrame:
    pipe = _counts_by_decile(df, country, "pipeline_total").rename(
        columns={"male": "pipe_male", "female": "pipe_female"}
    )

    if country == "Chile" and outcome in {"stem_first", "stem_any", "stem_admitted"}:
        chile_app = _load_chile_decile_applicant_denominators()
        if not chile_app.empty:
            pipe = pipe[["decile"]].merge(chile_app, on="decile", how="left")

    out = _counts_by_decile(df, country, outcome).rename(
        columns={"male": "out_male", "female": "out_female"}
    )
    m = pipe.merge(out, on="decile", how="inner")

    m = _aggregate_bottom_half_keep_top_deciles(m, "pipe_male", "pipe_female").merge(
        _aggregate_bottom_half_keep_top_deciles(m, "out_male", "out_female"),
        on="group",
        how="inner",
    )
    m["prob_male_pct"] = m["out_male"] / m["pipe_male"] * 100
    m["prob_female_pct"] = m["out_female"] / m["pipe_female"] * 100
    m["gap_female_minus_male_pp"] = m["prob_female_pct"] - m["prob_male_pct"]
    return m.rename(columns={"group": "score_bin", "pipe_male": "pipeline_male",
                 "pipe_female": "pipeline_female", "out_male": "outcome_male",
                 "out_female": "outcome_female"})[
        [
            "score_bin",
            "pipeline_male",
            "pipeline_female",
            "outcome_male",
            "outcome_female",
            "prob_male_pct",
            "prob_female_pct",
            "gap_female_minus_male_pp",
        ]
    ]


def _annotate(ax, bars, is_gap=False):
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        if is_gap:
            if h >= 0:
                y = min(h + 0.6, 43.2)
                va = "bottom"
            else:
                y = max(h - 0.8, -43.2)
                va = "top"
            txt = f"{h:.0f}"
        else:
            y = h + 0.8
            va = "bottom"
            txt = f"{h:.0f}%"
        ax.text(b.get_x() + b.get_width() / 2, y, txt, ha="center", va=va, fontsize=VAL_FS)


def _placeholder_panel(ax, title: str, message: str) -> None:
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xticks(GROUP_POS)
    ax.set_xticklabels(GROUP_LABELS, fontsize=TICK_FS)
    ax.set_xlim(0.5, len(GROUP_LABELS) + 0.5)
    ax.set_ylim(-30, 30)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=10)


def _collect(df: pd.DataFrame, per_country) -> pd.DataFrame:
    """Stack one country-level table per available country, labelled by country.

    Countries absent from the harmonized master are simply left out; the plot
    loop draws its placeholder panel when it finds no rows for one.
    """
    rows = []
    for country in COUNTRY_ORDER_10:
        source_country = DATA_COUNTRY_MAP.get(country, country)
        if source_country not in df["country"].unique():
            continue
        t = per_country(source_country)
        t.insert(0, "country", country)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def figure_a(df: pd.DataFrame) -> None:
    filename, figure = FIGURE_DEST["female_share"]
    published = fd.publish(
        _collect(df, lambda c: _female_share_by_decile(df, c)),
        filename=filename,
        figure=figure,
    )

    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharey=True)
    axes = axes.flatten()

    for ax, country in zip(axes, COUNTRY_ORDER_10):
        t = published[published["country"] == country]
        if t.empty:
            _placeholder_panel(ax, country, "Data unavailable\nin harmonized master")
            ax.set_ylim(0, 100)
            ax.axhline(50, color="black", linestyle="--", linewidth=1)
            continue

        x = np.array(GROUP_POS)
        y = t["female_share_pct"].to_numpy()
        bars = ax.bar(x, y, color=BAR_COLOR_SHARE, edgecolor="black", linewidth=0.6)
        _annotate(ax, bars, is_gap=False)
        ax.axhline(50, color="black", linestyle="--", linewidth=1)
        ax.set_title(country, fontsize=TITLE_FS)
        ax.set_xticks(GROUP_POS)
        ax.set_xticklabels(GROUP_LABELS, fontsize=TICK_FS)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)

    for i, ax in enumerate(axes):
        if i >= len(COUNTRY_ORDER_10):
            ax.axis("off")
            continue
        if i % 3 == 0:
            ax.set_ylabel("Female share (%)", fontsize=AX_LABEL_FS)
        if i >= 7:
            ax.set_xlabel("Score bin", fontsize=AX_LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "online_appendix_figureA_female_shares_by_decile.pdf")
    fig.savefig(OUT_DIR / "online_appendix_figureA_female_shares_by_decile.png", dpi=300)
    plt.close(fig)


def _gap_figure(df: pd.DataFrame, outcome: str, out_pdf: str) -> None:
    filename, figure = FIGURE_DEST[outcome]
    published = fd.publish(
        _collect(df, lambda c: _gap_by_decile(df, c, outcome)),
        filename=filename,
        figure=figure,
    )

    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharey=True)
    axes = axes.flatten()

    for ax, country in zip(axes, COUNTRY_ORDER_10):
        t = published[published["country"] == country]
        if t.empty:
            _placeholder_panel(ax, country, "Data unavailable\nin harmonized master")
            continue

        x = np.array(GROUP_POS)
        y = t["gap_female_minus_male_pp"].to_numpy()
        colors = [BAR_COLOR_POS if v >= 0 else BAR_COLOR_NEG for v in y]
        bars = ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.6)
        _annotate(ax, bars, is_gap=True)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(country, fontsize=TITLE_FS)
        ax.set_xticks(GROUP_POS)
        ax.set_xticklabels(GROUP_LABELS, fontsize=TICK_FS)
        ax.set_ylim(-44, 44)
        ax.grid(axis="y", alpha=0.2)

    for i, ax in enumerate(axes):
        if i >= len(COUNTRY_ORDER_10):
            ax.axis("off")
            continue
        if i % 3 == 0:
            ax.set_ylabel("Female minus male (pp)", fontsize=AX_LABEL_FS)
        if i >= 7:
            ax.set_xlabel("Score bin", fontsize=AX_LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)

    fig.tight_layout()
    fig.savefig(OUT_DIR / out_pdf)
    fig.savefig(OUT_DIR / out_pdf.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT)
    df = _select_preferred(df)

    figure_a(df)
    _gap_figure(df, "stem_first", "online_appendix_figureB_stem_first_gap_by_decile.pdf")
    _gap_figure(df, "stem_any", "online_appendix_figureC_stem_any_gap_by_decile.pdf")
    _gap_figure(df, "stem_admitted", "online_appendix_figureD_stem_admitted_gap_by_decile.pdf")

    fd.write_column_dictionary()
    print(f"Wrote Online Appendix figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
