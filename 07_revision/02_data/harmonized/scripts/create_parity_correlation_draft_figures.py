#!/usr/bin/env python3
"""Create draft parity-correlation figures for manuscript consideration.

Outputs:
- draft_figure5_wage_parity_pipeline_choice.(png|pdf), the main-text Figure 4
- draft_parity_pipeline_gap_vs_indexes.(png|pdf)
- draft_parity_stemfirst_gap_vs_indexes.(png|pdf)

Figures are top-10 based and mirror the style of manuscript cross-country
scatter correlations with country labels and fitted lines.

The numbers plotted are published to and read back from
``figure_data/figure_04_wage_parity_vs_gaps.csv`` via ``figure_data.py``. This is
the one figure whose x axis does not come from the harmonized admissions data:
the wage and index columns are external, so the published CSV is also the only
place the two sources are joined.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_data as fd


BASE = Path(__file__).resolve().parents[2]
FIG_INPUTS = BASE / "harmonized" / "master" / "figure_inputs_top10_summary.csv"
WEF_INPUT = BASE.parents[1] / "05_data" / "wef_gender_gap_2025_sample.csv"
WAGE_INPUT = BASE / "gender_wage_gap_10countries_mixed_sources.csv"
CHILE_DENOM_CACHE = BASE / "harmonized" / "cache" / "chile_applicant_denominators.csv"
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


def _chile_applicant_top10() -> dict[str, int] | None:
    """Chile's top-decile counts restricted to students who submitted an application.

    Same committed snapshot the main-text choice figures read; see that file's
    header for provenance.
    """
    if not CHILE_DENOM_CACHE.exists():
        print("  [chile] applicant denominators not found; parity panel will use "
              "score-holder denominators and will not match the choice figure")
        return None
    c = pd.read_csv(CHILE_DENOM_CACHE, comment="#").set_index("score_group")
    if "Decile 10" not in c.index:
        return None
    r = c.loc["Decile 10"]
    return {"male": int(r["male"]), "female": int(r["female"])}


def _load_inputs() -> pd.DataFrame:
    fig = pd.read_csv(FIG_INPUTS)
    fig = fig[fig["country"].isin(PREFERRED_VARIANT.keys())].copy()
    fig = fig[
        fig.apply(
            lambda r: r["sample_variant"] == PREFERRED_VARIANT.get(r["country"]),
            axis=1,
        )
    ].copy()

    # Top-10 pipeline gap in percentage points: female share minus male share.
    fig["pipeline_gap_female_minus_male_pp"] = (2.0 * fig["share_female_top10"] - 1.0) * 100.0
    # Already in proportion units in source table.
    fig["stem_first_gap_female_minus_male_pp"] = (
        fig["gender_gap_stem_first_top10_female_minus_male"] * 100.0
    )

    # Chile is the one setting where the decile base and the choice denominator
    # differ: deciles are built on all PSU score holders, but only about half of
    # them submit an application. The main-text choice figure conditions Chile's
    # probabilities on applicants, so this panel must do the same or the same
    # quantity appears with two different values in one paper. The pipeline gap
    # above is deliberately left on the full score-holder distribution, matching
    # the pipeline figure.
    ch = _chile_applicant_top10()
    if ch is not None:
        m = fig["country"] == "Chile"
        if m.any():
            r = fig.loc[m].iloc[0]
            gap = (
                r["n_stem_first_top10_female"] / ch["female"]
                - r["n_stem_first_top10_male"] / ch["male"]
            ) * 100.0
            was = r["gender_gap_stem_first_top10_female_minus_male"] * 100.0
            fig.loc[m, "stem_first_gap_female_minus_male_pp"] = gap
            print(f"  [chile] choice gap conditioned on applicants: {gap:.2f} pp (was {was:.2f})")

    wef = pd.read_csv(WEF_INPUT)
    wef = wef.rename(columns={"country_full": "country"})[
        ["country", "wef_gggi_2025_overall"]
    ]

    wage = pd.read_csv(WAGE_INPUT)
    wage = wage.rename(columns={"gender_wage_gap_percent": "wage_gap_percent"})[
        ["country", "wage_gap_percent"]
    ]
    # Convert wage gap (%) to a parity index where higher means more parity.
    wage["wage_parity_index"] = 1.0 - wage["wage_gap_percent"] / 100.0

    out = (
        fig[[
            "country",
            "pipeline_gap_female_minus_male_pp",
            "stem_first_gap_female_minus_male_pp",
        ]]
        .merge(wef, on="country", how="left")
        .merge(wage, on="country", how="left")
    )

    out["country"] = pd.Categorical(out["country"], categories=COUNTRY_ORDER, ordered=True)
    out = out.sort_values("country").reset_index(drop=True)
    return fd.publish(
        out,
        filename="figure_04_wage_parity_vs_gaps.csv",
        figure="Figure 4",
    )


def _scatter_with_fit(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    color: str,
    panel_title: str,
    x_limits: tuple[float, float] | None = None,
) -> None:
    d = df[["country", x_col, y_col]].dropna().copy()

    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    ax.scatter(x, y, s=52, color=color, edgecolor="black", linewidth=0.5, zorder=3)

    # The figure is drawn on a 12.6in canvas but placed at ~0.98\textwidth, so
    # on-page type is roughly a third of the size set here.
    for _, row in d.iterrows():
        ax.annotate(
            row["country"],
            (row[x_col], row[y_col]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=12,
        )

    if len(d) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = slope * xs + intercept
        ax.plot(xs, ys, color="0.25", linestyle="-", linewidth=1.4, zorder=2)
        ax.text(
            0.02,
            0.95,
            f"beta = {slope:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.7", "pad": 2.0},
        )

    ax.axhline(0, color="0.45", linestyle="--", linewidth=1.0)
    ax.grid(axis="both", alpha=0.2)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(panel_title, fontsize=11)


def _make_two_panel_figure(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    out_stem: str,
    fig_title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)

    _scatter_with_fit(
        axes[0],
        df,
        x_col="wef_gggi_2025_overall",
        y_col=y_col,
        x_label="WEF Global Gender Gap Index (2025, overall)",
        y_label=y_label,
        color="#5627FF",
        panel_title="Panel A: WEF parity index",
    )

    _scatter_with_fit(
        axes[1],
        df,
        x_col="wage_parity_index",
        y_col=y_col,
        x_label="Wage parity index (1 - gender wage gap)",
        y_label=y_label,
        color="#5627FF",
        panel_title="Panel B: Wage-based parity index",
        x_limits=(0.5, 1.0),
    )

    fig.suptitle(fig_title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{out_stem}.pdf")
    plt.close(fig)


def _make_figure5_wage_only(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

    _scatter_with_fit(
        axes[0],
        df,
        x_col="wage_parity_index",
        y_col="pipeline_gap_female_minus_male_pp",
        x_label="Wage parity index (1 - gender wage gap)",
        y_label="Pipeline gap (female - male, pp)",
        color="#5627FF",
        panel_title="Panel A: Pipeline gap",
        x_limits=(0.5, 1.0),
    )

    _scatter_with_fit(
        axes[1],
        df,
        x_col="wage_parity_index",
        y_col="stem_first_gap_female_minus_male_pp",
        x_label="Wage parity index (1 - gender wage gap)",
        y_label="STEM first-choice gap (female - male, pp)",
        color="#5627FF",
        panel_title="Panel B: Choice gap",
        x_limits=(0.5, 1.0),
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "draft_figure5_wage_parity_pipeline_choice.png", dpi=300)
    fig.savefig(OUT_DIR / "draft_figure5_wage_parity_pipeline_choice.pdf")
    plt.close(fig)


def main() -> None:
    data = _load_inputs()

    _make_two_panel_figure(
        data,
        y_col="pipeline_gap_female_minus_male_pp",
        y_label="Pipeline gap (female - male, pp)",
        out_stem="draft_parity_pipeline_gap_vs_indexes",
        fig_title="Top-10 Pipeline Gap vs Gender Parity Indexes",
    )

    _make_two_panel_figure(
        data,
        y_col="stem_first_gap_female_minus_male_pp",
        y_label="STEM first-choice gap (female - male, pp)",
        out_stem="draft_parity_stemfirst_gap_vs_indexes",
        fig_title="Top-10 STEM First-Choice Gap vs Gender Parity Indexes",
    )

    _make_figure5_wage_only(data)

    fd.write_column_dictionary()
    print("Created draft figures in:", OUT_DIR)


if __name__ == "__main__":
    main()
