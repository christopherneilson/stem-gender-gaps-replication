#!/usr/bin/env python3
"""Generate figures/tables for referee-requested robustness and extensions.

This script uses harmonized_counts_long.csv and produces:
1) Top-10 outcomes beyond first choice (stem_any, stem_first, stem_admitted)
   - Gender composition among outcome participants
   - Conditional probabilities by sex within top-10 pipeline
2) Threshold robustness (Top 5/10/20/30) for application and admission gaps
3) Bottom-90 context table (deciles 1-9) with male/female STEM rates by country
4) High-return non-STEM benchmark: STEM vs Business/Law gaps in top decile
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import reproducible  # noqa: F401  (sets SOURCE_DATE_EPOCH on import)
from canonical_variants import PREFERRED_VARIANT, COUNTRY_ORDER

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "harmonized" / "master" / "harmonized_counts_long.csv"
OUT_DIR = BASE / "harmonized" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


THRESHOLDS = {
    # Greece has no top-1% row -- its data arrive by decile and top 5% -- so the
    # 1% threshold covers nine settings and Greece drops out of that column.
    1: [("top_percent", 1)],
    5: [("top_percent", 5)],
    10: [("decile", 10)],
    20: [("decile", 9), ("decile", 10)],
    30: [("decile", 8), ("decile", 9), ("decile", 10)],
}

MALE_COLOR = "#FFC20A"
FEMALE_COLOR = "#0C7BDC"
STEM_COLOR = "#B03A2E"
BUS_COLOR = "#1F77B4"


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


def _pivot_counts(df: pd.DataFrame, outcome: str, groups: list[tuple[str, float]], keep_all: bool = False) -> pd.DataFrame:
    d = _subset_groups(df[df["outcome"] == outcome], groups)
    if d.empty:
        return pd.DataFrame(columns=["country", "male", "female"])
    # keep_all keeps every setting in the grid, so a cell a setting cannot form
    # (Greece at the top 1%) stays as a zero-count row that downstream code marks
    # unavailable; otherwise such settings are dropped. Explicit because pandas 3
    # flips the default of `observed`.
    w = d.pivot_table(index=["country"], columns="sex", values="n", aggfunc="sum", observed=not keep_all).reset_index()
    w.columns.name = None
    for col in ["male", "female"]:
        if col not in w.columns:
            w[col] = np.nan
    return w[["country", "male", "female"]]


def top10_three_outcomes(df: pd.DataFrame) -> None:
    groups = THRESHOLDS[10]
    outcomes = ["stem_any", "stem_first", "stem_admitted"]

    rows = []
    for outcome in outcomes:
        pipe = _pivot_counts(df, "pipeline_total", groups).rename(
            columns={"male": "pipeline_male", "female": "pipeline_female"}
        )
        num = _pivot_counts(df, outcome, groups).rename(columns={"male": "num_male", "female": "num_female"})
        merged = pipe.merge(num, on="country", how="inner")
        merged["outcome"] = outcome

        merged["prob_male"] = merged["num_male"] / merged["pipeline_male"]
        merged["prob_female"] = merged["num_female"] / merged["pipeline_female"]
        merged["gap_female_minus_male"] = merged["prob_female"] - merged["prob_male"]

        tot = merged["num_female"] + merged["num_male"]
        merged["female_share_among_outcome"] = merged["num_female"] / tot
        merged["male_share_among_outcome"] = merged["num_male"] / tot
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out["country"] = pd.Categorical(out["country"], categories=COUNTRY_ORDER, ordered=True)
    out = out.sort_values(["country", "outcome"]).reset_index(drop=True)

    out_csv = OUT_DIR / "top10_three_outcomes_metrics.csv"
    out.to_csv(out_csv, index=False)

    # Figure A: female share among outcome participants (similar to Figure 1 style target).
    fig, ax = plt.subplots(figsize=(12, 6.8))
    x = np.arange(len(COUNTRY_ORDER))
    width = 0.23
    labels = {
        "stem_any": "Any STEM in list",
        "stem_first": "STEM first",
        "stem_admitted": "Admitted STEM",
    }
    colors = {
        "stem_any": "#4C78A8",
        "stem_first": "#E45756",
        "stem_admitted": "#72B7B2",
    }

    for i, outcome in enumerate(outcomes):
        tmp = out[out["outcome"] == outcome].set_index("country").reindex(COUNTRY_ORDER)
        y = (tmp["female_share_among_outcome"] * 100).to_numpy()
        bars = ax.bar(x + (i - 1) * width, y, width=width, label=labels[outcome],
                      color=colors[outcome], edgecolor="black", linewidth=0.7)
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(COUNTRY_ORDER)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Female share among students in outcome (%)")
    ax.set_title("Top 10%: Gender Shares by STEM Outcome")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top10_gender_shares_three_outcomes.png", dpi=300)
    fig.savefig(OUT_DIR / "top10_gender_shares_three_outcomes.pdf")

    # Figure B: conditional probabilities by sex in top 10%, split by outcome panels.
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
    for ax, outcome in zip(axes, outcomes):
        tmp = out[out["outcome"] == outcome].set_index("country").reindex(COUNTRY_ORDER)
        male = (tmp["prob_male"] * 100).to_numpy()
        female = (tmp["prob_female"] * 100).to_numpy()
        xx = np.arange(len(COUNTRY_ORDER))
        w = 0.38
        ax.bar(xx - w / 2, male, width=w, color=MALE_COLOR, edgecolor="black", linewidth=0.7, label="Male")
        ax.bar(xx + w / 2, female, width=w, color=FEMALE_COLOR, edgecolor="black", linewidth=0.7, label="Female")
        ax.set_xticks(xx)
        ax.set_xticklabels(COUNTRY_ORDER, rotation=45, ha="right")
        ax.set_title(labels[outcome])
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Probability within top 10% (%)")
    axes[0].set_ylim(0, 100)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig2.legend(handles, labels_, loc="upper center", ncol=2, frameon=True, edgecolor="black")
    fig2.suptitle("Top 10%: STEM Outcomes by Sex")
    fig2.tight_layout(rect=(0, 0, 1, 0.92))
    fig2.savefig(OUT_DIR / "top10_choice_probabilities_three_outcomes.png", dpi=300)
    fig2.savefig(OUT_DIR / "top10_choice_probabilities_three_outcomes.pdf")


def threshold_robustness(df: pd.DataFrame) -> None:
    outcomes = ["stem_first", "stem_any", "stem_admitted"]
    rows = []

    for thr, groups in THRESHOLDS.items():
        pipe = _pivot_counts(df, "pipeline_total", groups, keep_all=True).rename(
            columns={"male": "pipeline_male", "female": "pipeline_female"}
        )
        for outcome in outcomes:
            num = _pivot_counts(df, outcome, groups, keep_all=True).rename(columns={"male": "num_male", "female": "num_female"})
            merged = pipe.merge(num, on="country", how="inner")
            merged["threshold"] = thr
            merged["outcome"] = outcome
            merged["prob_male"] = merged["num_male"] / merged["pipeline_male"]
            merged["prob_female"] = merged["num_female"] / merged["pipeline_female"]
            merged["gap_female_minus_male"] = merged["prob_female"] - merged["prob_male"]
            rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out["country"] = pd.Categorical(out["country"], categories=COUNTRY_ORDER, ordered=True)
    out = out.sort_values(["outcome", "country", "threshold"]).reset_index(drop=True)
    out.to_csv(OUT_DIR / "threshold_robustness_by_country.csv", index=False, lineterminator="\n")

    pooled = (
        out.groupby(["threshold", "outcome"], as_index=False)
        .agg(
            pipeline_male=("pipeline_male", "sum"),
            pipeline_female=("pipeline_female", "sum"),
            num_male=("num_male", "sum"),
            num_female=("num_female", "sum"),
            n_countries=("pipeline_male", lambda s: int((s.fillna(0) > 0).sum())),  # settings with a group
        )
    )
    pooled["prob_male"] = pooled["num_male"] / pooled["pipeline_male"]
    pooled["prob_female"] = pooled["num_female"] / pooled["pipeline_female"]
    pooled["gap_female_minus_male"] = pooled["prob_female"] - pooled["prob_male"]
    pooled.to_csv(OUT_DIR / "threshold_robustness_pooled.csv", index=False, lineterminator="\n")

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    style = {
        "stem_first": ("STEM first", "#E45756", "o"),
        "stem_any": ("Any STEM", "#4C78A8", "s"),
        "stem_admitted": ("Admitted STEM", "#72B7B2", "^"),
    }
    for outcome in outcomes:
        tmp = pooled[pooled["outcome"] == outcome].sort_values("threshold")
        label, color, marker = style[outcome]
        ax.plot(tmp["threshold"], tmp["gap_female_minus_male"] * 100,
                color=color, marker=marker, linewidth=2, label=label)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([5, 10, 20, 30])
    ax.set_xlabel("Top score threshold (%)")
    ax.set_ylabel("Female minus male probability (pp)")
    ax.set_title("Threshold Robustness: Gender Gaps in Applications and Admission")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "threshold_robustness_choice_admission.png", dpi=300)
    fig.savefig(OUT_DIR / "threshold_robustness_choice_admission.pdf")


def bottom90_context_table(df: pd.DataFrame) -> None:
    deciles = [("decile", i) for i in range(1, 10)]
    pipe = _subset_groups(df[df["outcome"] == "pipeline_total"], deciles)
    stem_any = _subset_groups(df[df["outcome"] == "stem_any"], deciles)
    stem_first = _subset_groups(df[df["outcome"] == "stem_first"], deciles)

    key_cols = ["country", "score_group_label", "sex"]
    p = pipe.groupby(key_cols, as_index=False)["n"].sum().rename(columns={"n": "n_pipeline"})
    a = stem_any.groupby(key_cols, as_index=False)["n"].sum().rename(columns={"n": "n_stem_any"})
    f = stem_first.groupby(key_cols, as_index=False)["n"].sum().rename(columns={"n": "n_stem_first"})

    merged = p.merge(a, on=key_cols, how="left").merge(f, on=key_cols, how="left")
    merged[["n_stem_any", "n_stem_first"]] = merged[["n_stem_any", "n_stem_first"]].fillna(0)
    merged["rate_stem_any"] = merged["n_stem_any"] / merged["n_pipeline"]
    merged["rate_stem_first"] = merged["n_stem_first"] / merged["n_pipeline"]

    wide = merged.pivot_table(
        index=["country", "score_group_label"],
        columns="sex",
        values=["rate_stem_any", "rate_stem_first"],
        aggfunc="first",
    ).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns.to_flat_index()]

    wide["gap_any_pp_female_minus_male"] = (wide["rate_stem_any_female"] - wide["rate_stem_any_male"]) * 100
    wide["gap_first_pp_female_minus_male"] = (wide["rate_stem_first_female"] - wide["rate_stem_first_male"]) * 100

    wide["decile"] = wide["score_group_label"].str.extract(r"(\d+)").astype(int)
    wide["country"] = pd.Categorical(wide["country"], categories=COUNTRY_ORDER, ordered=True)
    wide = wide.sort_values(["country", "decile"]).reset_index(drop=True)

    out_csv = OUT_DIR / "bottom90_decile_country_stem_rates.csv"
    wide.to_csv(out_csv, index=False)

    # LaTeX table with percentages for manuscript integration.
    tbl = wide.copy()
    for col in [
        "rate_stem_any_male", "rate_stem_any_female", "rate_stem_first_male", "rate_stem_first_female"
    ]:
        tbl[col] = (tbl[col] * 100).map(lambda x: f"{x:.1f}")
    for col in ["gap_any_pp_female_minus_male", "gap_first_pp_female_minus_male"]:
        tbl[col] = tbl[col].map(lambda x: f"{x:.1f}")

    tbl = tbl[
        [
            "country",
            "decile",
            "rate_stem_any_male",
            "rate_stem_any_female",
            "gap_any_pp_female_minus_male",
            "rate_stem_first_male",
            "rate_stem_first_female",
            "gap_first_pp_female_minus_male",
        ]
    ]
    tbl = tbl.rename(
        columns={
            "country": "Country",
            "decile": "Decile",
            "rate_stem_any_male": "Any STEM male (%)",
            "rate_stem_any_female": "Any STEM female (%)",
            "gap_any_pp_female_minus_male": "Any STEM gap (pp)",
            "rate_stem_first_male": "STEM first male (%)",
            "rate_stem_first_female": "STEM first female (%)",
            "gap_first_pp_female_minus_male": "STEM first gap (pp)",
        }
    )

    latex = tbl.to_latex(index=False, escape=True, column_format="llrrrrrr")
    (OUT_DIR / "bottom90_decile_country_stem_rates.tex").write_text(latex, encoding="utf-8")


def top10_stem_vs_business_benchmark(df: pd.DataFrame) -> None:
    groups = THRESHOLDS[10]
    rows = []

    comparisons = [
        ("stem_first", "business_first", "First-choice"),
        ("stem_any", "business_any", "Any in list"),
        ("stem_admitted", "business_admitted", "Admitted"),
    ]

    pipe = _pivot_counts(df, "pipeline_total", groups).rename(
        columns={"male": "pipeline_male", "female": "pipeline_female"}
    )

    for stem_outcome, bus_outcome, label in comparisons:
        stem = _pivot_counts(df, stem_outcome, groups).rename(columns={"male": "stem_male", "female": "stem_female"})
        bus = _pivot_counts(df, bus_outcome, groups).rename(columns={"male": "bus_male", "female": "bus_female"})
        merged = pipe.merge(stem, on="country", how="inner").merge(bus, on="country", how="inner")

        merged["stem_gap_pp"] = ((merged["stem_female"] / merged["pipeline_female"]) - (merged["stem_male"] / merged["pipeline_male"])) * 100
        merged["business_gap_pp"] = ((merged["bus_female"] / merged["pipeline_female"]) - (merged["bus_male"] / merged["pipeline_male"])) * 100
        merged["comparison"] = label
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out["country"] = pd.Categorical(out["country"], categories=COUNTRY_ORDER, ordered=True)
    out = out.sort_values(["comparison", "country"]).reset_index(drop=True)
    out.to_csv(OUT_DIR / "top10_stem_vs_business_benchmark.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
    for ax, label in zip(axes, ["First-choice", "Any in list", "Admitted"]):
        tmp = out[out["comparison"] == label].set_index("country").reindex(COUNTRY_ORDER)
        x = np.arange(len(COUNTRY_ORDER))
        w = 0.36
        ax.bar(x - w / 2, tmp["stem_gap_pp"], width=w, color=STEM_COLOR, edgecolor="black", linewidth=0.7, label="STEM gap")
        ax.bar(x + w / 2, tmp["business_gap_pp"], width=w, color=BUS_COLOR, edgecolor="black", linewidth=0.7, label="Business/Law gap")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(COUNTRY_ORDER, rotation=45, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Female minus male probability (pp), top 10%")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=True, edgecolor="black")
    fig.suptitle("Top 10% Benchmark: STEM vs Business/Law Gender Gaps")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "top10_stem_vs_business_benchmark.png", dpi=300)
    fig.savefig(OUT_DIR / "top10_stem_vs_business_benchmark.pdf")


def main() -> None:
    df = pd.read_csv(INPUT)
    df = _select_preferred(df)

    top10_three_outcomes(df)
    threshold_robustness(df)
    bottom90_context_table(df)
    top10_stem_vs_business_benchmark(df)

    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
