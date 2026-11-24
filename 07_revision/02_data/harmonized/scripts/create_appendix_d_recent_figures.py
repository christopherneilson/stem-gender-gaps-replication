#!/usr/bin/env python3
"""Create Appendix D figures for the cohort closest to 2018 by country.

Given the currently harmonized dataset structure, countries are selected as follows:
- Exact 2018 cohorts where available (China, Finland).
- Closest available single-year cohorts where provided in source variants (Sweden 2017, Taiwan 2010, Brazil 2016).
- Closest available pooled windows when single-year cohorts are not available in harmonized rows.

The panels are drawn from ``figure_data/figure_D01_recent_cohort_top10_gaps.csv``,
written and read back by ``figure_data.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import figure_data as fd

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "harmonized" / "master" / "harmonized_counts_long.csv"
OUT_DIR = BASE / "harmonized" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHILE_CANDIDATE_DIRS = [
    Path("confidential"),
    BASE,
]
CHILE_2018_CACHE = BASE / "harmonized" / "cache" / "chile_2018_top10_cached.csv"

COUNTRY_ORDER = [
    "Australia",
    "Brazil",
    "Chile",
    "China",
    "Finland",
    "Greece",
    "Spain",
    "Sweden",
    "Taiwan",
    "Uganda",
]

# Cohort selection closest to 2018 within currently harmonized source variants.
SELECTION = {
    "Australia": {"sample_variant": "atar_2010", "source_file_contains": None, "years_used": "2010"},
    "Brazil": {"sample_variant": "reply_tables", "source_file_contains": None, "years_used": "2016"},
    "Chile": {"sample_variant": "chile_2012_2018", "source_file_contains": None, "years_used": "2012--2018 (pooled)"},
    "China": {"sample_variant": "china_ningxia_2018", "source_file_contains": None, "years_used": "2018"},
    "Finland": {"sample_variant": "year_2018", "source_file_contains": None, "years_used": "2018"},
    "Greece": {"sample_variant": "years_2003_2012", "source_file_contains": None, "years_used": "2003--2012 (pooled)"},
    "Spain": {"sample_variant": "esp_2020_2024_tables", "source_file_contains": "_2020", "years_used": "2020"},
    "Sweden": {"sample_variant": "reply_docx_2017", "source_file_contains": None, "years_used": "2017"},
    "Taiwan": {"sample_variant": "reply_pdf_2010", "source_file_contains": None, "years_used": "2010"},
    "Uganda": {"sample_variant": "reply_docx", "source_file_contains": None, "years_used": "2017/2018"},
}

OUTCOMES = {
    "pipeline_total": "Pipeline gap (female share minus male share among top 10%, pp)",
    "stem_first": "STEM first-choice gap (female minus male probability in top 10%, pp)",
    "stem_any": "Any-STEM gap (female minus male probability in top 10%, pp)",
}


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for country, cfg in SELECTION.items():
        s = df[(df["country"] == country) & (df["sample_variant"] == cfg["sample_variant"])].copy()
        sf = cfg["source_file_contains"]
        if sf is not None:
            s = s[s["source_file"].astype(str).str.contains(sf, na=False)]
        keep.append(s)
    out = pd.concat(keep, ignore_index=True)
    return out


def top10_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["score_group_type"] == "decile") & (df["score_group_value"] == 10)].copy()


def pivot_counts(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    s = df[df["outcome"] == outcome].copy()
    w = s.groupby(["country", "sex"], as_index=False)["n"].sum().pivot(index="country", columns="sex", values="n")
    w = w.reset_index()
    if "female" not in w.columns:
        w["female"] = 0
    if "male" not in w.columns:
        w["male"] = 0
    return w[["country", "female", "male"]]


def compute_gaps(df: pd.DataFrame) -> pd.DataFrame:
    pipe = pivot_counts(df, "pipeline_total").rename(columns={"female": "pipe_f", "male": "pipe_m"})

    # Pipeline gap as female share minus male share among top-10 students.
    pipe["gap_pipeline_pp"] = (pipe["pipe_f"] / (pipe["pipe_f"] + pipe["pipe_m"]) - pipe["pipe_m"] / (pipe["pipe_f"] + pipe["pipe_m"])) * 100

    first = pivot_counts(df, "stem_first").rename(columns={"female": "first_f", "male": "first_m"})
    anys = pivot_counts(df, "stem_any").rename(columns={"female": "any_f", "male": "any_m"})

    merged = pipe.merge(first, on="country", how="left").merge(anys, on="country", how="left")
    merged[["first_f", "first_m", "any_f", "any_m"]] = merged[["first_f", "first_m", "any_f", "any_m"]].fillna(0)

    merged["gap_stem_first_pp"] = (merged["first_f"] / merged["pipe_f"] - merged["first_m"] / merged["pipe_m"]) * 100
    merged["gap_stem_any_pp"] = (merged["any_f"] / merged["pipe_f"] - merged["any_m"] / merged["pipe_m"]) * 100

    merged["years_used"] = merged["country"].map({k: v["years_used"] for k, v in SELECTION.items()})
    merged["country"] = pd.Categorical(merged["country"], categories=COUNTRY_ORDER, ordered=True)
    merged = merged.sort_values("country").reset_index(drop=True)

    # Override Chile with exact 2018 top-10 counts when raw yearly summary files are available.
    ch = _chile_2018_top10_counts()
    if ch is not None:
        m = merged["country"] == "Chile"
        if m.any():
            merged.loc[m, "pipe_f"] = ch["pipe_f"]
            merged.loc[m, "pipe_m"] = ch["pipe_m"]
            merged.loc[m, "first_f"] = ch["first_f"]
            merged.loc[m, "first_m"] = ch["first_m"]
            merged.loc[m, "any_f"] = ch["any_f"]
            merged.loc[m, "any_m"] = ch["any_m"]
            merged.loc[m, "gap_pipeline_pp"] = (ch["pipe_f"] / (ch["pipe_f"] + ch["pipe_m"]) - ch["pipe_m"] / (ch["pipe_f"] + ch["pipe_m"])) * 100
            merged.loc[m, "gap_stem_first_pp"] = (ch["first_f"] / ch["pipe_f"] - ch["first_m"] / ch["pipe_m"]) * 100
            merged.loc[m, "gap_stem_any_pp"] = (ch["any_f"] / ch["pipe_f"] - ch["any_m"] / ch["pipe_m"]) * 100
            merged.loc[m, "years_used"] = "2018"

    return merged


def _first_existing_path(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _chile_2018_cached_counts(reason: str) -> dict[str, float] | None:
    """Fall back to a committed snapshot of Chile's 2018 top-10 counts.

    The yearly summaries are written by 04_chile/create_data_chile.do into a local
    Dropbox folder that only some coauthors have mounted. Without this fallback the
    override below silently fails and Chile reverts to the pooled 2012--2018 window,
    which is the main-text sample and so contributes nothing to this appendix.
    """
    if not CHILE_2018_CACHE.exists():
        print(f"  [chile] {reason}; no cached snapshot -- falling back to the pooled window")
        return None
    cached = pd.read_csv(CHILE_2018_CACHE, comment="#")
    if cached.empty:
        return None
    r = cached.iloc[0]
    print(f"  [chile] {reason}; loaded 2018 top-10 counts from {CHILE_2018_CACHE.name}")
    return {k: float(r[k]) for k in ("pipe_m", "pipe_f", "first_m", "first_f", "any_m", "any_f")}


def _chile_2018_top10_counts() -> dict[str, float] | None:
    base = _first_existing_path(CHILE_CANDIDATE_DIRS)
    if base is None:
        return _chile_2018_cached_counts("raw yearly summaries not found")

    f_pipe = base / "chile_pipeline_by_decile_gender.dta"
    f_out = base / "chile_outcomes_by_decile_gender.dta"
    if not f_pipe.exists() or not f_out.exists():
        return _chile_2018_cached_counts("raw yearly summaries not found")

    d_pipe = pd.read_stata(f_pipe, convert_categoricals=False)
    d_out = pd.read_stata(f_out, convert_categoricals=False)

    d_pipe = d_pipe[(d_pipe["year"] == 2018) & (d_pipe["score_decile"] == 10)]
    d_out = d_out[(d_out["year"] == 2018) & (d_out["score_decile"] == 10)]
    if d_pipe.empty or d_out.empty:
        return _chile_2018_cached_counts("raw summaries hold no year == 2018 top-decile rows")

    p = d_pipe.groupby("female", as_index=False)["n"].sum()
    o = d_out.groupby("female", as_index=False)[["stem_first", "stem_any"]].sum()
    m = p.merge(o, on="female", how="inner")

    male = m[m["female"] == 0]
    female = m[m["female"] == 1]
    if male.empty or female.empty:
        return _chile_2018_cached_counts("raw 2018 top-decile rows are missing a sex cell")

    return {
        "pipe_m": float(male["n"].iloc[0]),
        "pipe_f": float(female["n"].iloc[0]),
        "first_m": float(male["stem_first"].iloc[0]),
        "first_f": float(female["stem_first"].iloc[0]),
        "any_m": float(male["stem_any"].iloc[0]),
        "any_f": float(female["stem_any"].iloc[0]),
    }


# The working frame uses short internal names that also appear in the committed
# Chile cache, so they are renamed on the way out rather than in place: the
# published CSV then uses the same vocabulary as every other figure_data file.
D01_EXPORT_NAMES = {
    "pipe_f": "pipeline_female",
    "pipe_m": "pipeline_male",
    "first_f": "stem_first_female",
    "first_m": "stem_first_male",
    "any_f": "stem_any_female",
    "any_m": "stem_any_male",
    "gap_pipeline_pp": "pipeline_gap_female_minus_male_pp",
    "gap_stem_first_pp": "stem_first_gap_female_minus_male_pp",
    "gap_stem_any_pp": "stem_any_gap_female_minus_male_pp",
}


def save_outputs(g: pd.DataFrame) -> None:
    years_tbl = g[["country", "years_used"]].copy()
    years_tbl.to_csv(OUT_DIR / "appendix_d_recent_cohort_years_used.csv", index=False)

    # Published first, then plotted from the file that comes back, so the CSV a
    # coauthor opens is literally what the panels below draw.
    g = fd.publish(
        g.rename(columns=D01_EXPORT_NAMES),
        filename="figure_D01_recent_cohort_top10_gaps.csv",
        figure="Figure D.I",
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    # One colour for all three panels. Each panel is a separate measure named in
    # its own title, so colour carries no information here; giving the panels
    # three different colours would invite reading them as the sex encoding used
    # in the main-text figures, where navy is male and teal female.
    PANEL_COLOR = "#5627FF"
    series = [
        ("pipeline_gap_female_minus_male_pp", OUTCOMES["pipeline_total"], PANEL_COLOR),
        ("stem_first_gap_female_minus_male_pp", OUTCOMES["stem_first"], PANEL_COLOR),
        ("stem_any_gap_female_minus_male_pp", OUTCOMES["stem_any"], PANEL_COLOR),
    ]

    x = range(len(g))
    labels = g["country"].astype(str).tolist()

    for ax, (col, title, color) in zip(axes, series):
        vals = g[col].tolist()
        bars = ax.bar(x, vals, color=color, edgecolor="black", linewidth=0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel("pp")
        for b, v in zip(bars, vals):
            y = b.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = 0.4 if y >= 0 else -0.4
            ax.text(b.get_x() + b.get_width() / 2, y + offset, f"{v:.0f}", ha="center", va=va, fontsize=8)

    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(labels, rotation=30, ha="right")
    # in-figure title removed; the LaTeX \caption serves as the figure title
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_DIR / "appendix_d_recent_cohort_top10_gaps.pdf")
    fig.savefig(OUT_DIR / "appendix_d_recent_cohort_top10_gaps.png", dpi=300)


def main() -> None:
    d = pd.read_csv(INPUT)
    d = select_rows(d)
    d = top10_subset(d)
    gaps = compute_gaps(d)
    save_outputs(gaps)
    fd.write_column_dictionary()
    print(f"Wrote Appendix D cohort outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

