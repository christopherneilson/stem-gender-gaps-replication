#!/usr/bin/env python3
"""Publish the numbers behind each manuscript figure to ``harmonized/figure_data``.

Every figure script routes its plotting frame through :func:`publish` or
:func:`publish_grouped` here, which writes the frame to a CSV named for the
figure number as it appears in the paper and then reads it straight back. The
caller plots from the re-read copy.

The round trip is the point. An export written alongside a figure can drift from
it -- someone edits the plotting code, forgets the export, and the CSV quietly
starts describing a figure that no longer exists. Here the CSV *is* the figure's
input, so the two cannot disagree: if the file were wrong, the figure would be
wrong in exactly the same way.

A coauthor who wants to check or redraw a figure opens the file named for it and
needs nothing else. See ``figure_data/README.md`` for the figure-to-file map.
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
FIGURE_DATA_DIR = BASE / "figure_data"

# Panels that the scripts still produce but no manuscript \includegraphics
# points at (the admitted-outcome families, and the two superseded Figure 1
# variants). They are published too, so nothing is silently unavailable, but
# into a subfolder and tagged with this label so a coauthor cannot mistake one
# for a figure in the paper.
UNUSED_DIR = "not_in_manuscript"
UNUSED_LABEL = "not in manuscript"

UNIT_COUNT = "count of students"
UNIT_PCT = "percent"
UNIT_PP = "percentage points (female minus male)"
UNIT_LABEL = "label"
UNIT_INDEX = "index, 0-1"

# Every column published to figure_data must appear here, or publishing fails.
# That is deliberate: it is what makes "all columns are documented" a property
# the build enforces rather than a claim someone has to keep re-checking. Adding
# a column to a figure means adding its meaning here in the same edit.
#
# A name means the same thing in every file that carries it. Where two figures
# report a share of different denominators, the names differ rather than the
# meaning silently changing between files.
GLOSSARY: dict[str, tuple[str, str]] = {
    # Identifiers
    "figure": (UNIT_LABEL, "Manuscript exhibit this row belongs to"),
    "threshold": (UNIT_LABEL, "Performance group: Top 5%, Top 10% or Top 20%"),
    "score_bin": (UNIT_LABEL, "Score group: 1-5 (deciles 1-5 pooled), then 6 to 10"),
    "country": (UNIT_LABEL, "Educational system"),
    "outcome": (UNIT_LABEL, "Outcome code, defined in ../schema/outcome_reference.csv"),
    "years_used": (UNIT_LABEL, "Cohort year(s) behind this row"),
    # Headcounts
    "n_male": (UNIT_COUNT, "Men in the group"),
    "n_female": (UNIT_COUNT, "Women in the group"),
    "n_total": (UNIT_COUNT, "Men plus women in the group"),
    "pipeline_male": (UNIT_COUNT, "Men in the performance group (the denominator)"),
    "pipeline_female": (UNIT_COUNT, "Women in the performance group (the denominator)"),
    "outcome_male": (UNIT_COUNT, "Men with the outcome (the numerator)"),
    "outcome_female": (UNIT_COUNT, "Women with the outcome (the numerator)"),
    "stem_first_male": (UNIT_COUNT, "Men ranking a STEM program first"),
    "stem_first_female": (UNIT_COUNT, "Women ranking a STEM program first"),
    "stem_any_male": (UNIT_COUNT, "Men listing a STEM program anywhere"),
    "stem_any_female": (UNIT_COUNT, "Women listing a STEM program anywhere"),
    # Shares of a performance group, by sex (sums to 100 across the two sexes)
    "male_share_in_top_group_pct": (UNIT_PCT, "Men as a share of the performance group"),
    "female_share_in_top_group_pct": (UNIT_PCT, "Women as a share of the performance group"),
    # Shares of a set of applicants, by sex (sums to 100 across the two sexes)
    "male_share_among_applicants_pct": (
        UNIT_PCT,
        "Men as a share of students with this outcome in the performance group",
    ),
    "female_share_among_applicants_pct": (
        UNIT_PCT,
        "Women as a share of students with this outcome in the performance group",
    ),
    # Probabilities within a sex (does not sum to 100 across the two sexes)
    "prob_male_pct": (UNIT_PCT, "Share of men in the group with the outcome"),
    "prob_female_pct": (UNIT_PCT, "Share of women in the group with the outcome"),
    # Gaps
    "gap_female_minus_male_pp": (UNIT_PP, "prob_female_pct minus prob_male_pct"),
    "pipeline_gap_female_minus_male_pp": (
        UNIT_PP,
        "Pipeline gap: female minus male share of the performance group",
    ),
    "stem_first_gap_female_minus_male_pp": (
        UNIT_PP,
        "Choice gap: female minus male probability of ranking STEM first",
    ),
    "stem_any_gap_female_minus_male_pp": (
        UNIT_PP,
        "Female minus male probability of listing STEM anywhere",
    ),
    # Figure 4 only: external, not from the harmonized admissions data
    "wage_gap_percent": (UNIT_PCT, "Gender wage gap (ILO or country source); external"),
    "wage_parity_index": (UNIT_INDEX, "1 minus the wage gap; 1 is parity; external"),
    "wef_gggi_2025_overall": (UNIT_INDEX, "WEF Global Gender Gap Index 2025; external"),
    # not_in_manuscript only
    "female_share_among_outcome_pct": (
        UNIT_PCT,
        "Women as a share of students with the outcome (superseded panels)",
    ),
    "female_share_pct": (UNIT_PCT, "Women as a share of the bar (superseded panels)"),
    "male_share_pct": (UNIT_PCT, "Men as a share of the bar (superseded panels)"),
    "num_male": (UNIT_COUNT, "Men with the outcome (superseded panels)"),
    "num_female": (UNIT_COUNT, "Women with the outcome (superseded panels)"),
}

COLUMN_DICTIONARY = "columns.csv"


def _write(frame: pd.DataFrame, filename: str, figure: str) -> Path:
    out = frame.copy()
    out.insert(0, "figure", figure)

    undocumented = [c for c in out.columns if c not in GLOSSARY]
    if undocumented:
        raise KeyError(
            f"{filename} publishes undocumented column(s) {undocumented}. "
            f"Add them to GLOSSARY in {Path(__file__).name} so figure_data/"
            f"{COLUMN_DICTIONARY} stays complete."
        )

    path = FIGURE_DATA_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def write_column_dictionary() -> Path:
    """Write the glossary to figure_data/columns.csv.

    Static, so every script writing it produces the same file. Call it at the
    end of main() to keep the dictionary next to the data it describes.
    """
    rows = [
        {"column": name, "unit": unit, "meaning": meaning}
        for name, (unit, meaning) in GLOSSARY.items()
    ]
    path = FIGURE_DATA_DIR / COLUMN_DICTIONARY
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def publish(frame: pd.DataFrame, *, filename: str, figure: str) -> pd.DataFrame:
    """Write one frame to one CSV and return it as read back from disk.

    ``figure`` is the manuscript exhibit number ("Figure 2", "Figure B.IV") and
    becomes the first column, so the file identifies itself even once it has
    been renamed, emailed, or pasted into a spreadsheet.
    """
    return pd.read_csv(_write(frame, filename, figure))


def publish_grouped(
    frames: dict[str, pd.DataFrame],
    destinations: dict[str, tuple[str, str]],
    *,
    key: str,
) -> dict[str, pd.DataFrame]:
    """Publish frames that share a destination CSV, keyed by e.g. threshold.

    ``destinations`` maps each key of ``frames`` to ``(filename, figure)``. Keys
    routed to the same filename are concatenated in ``frames`` order into a
    single file: one figure of the paper is one CSV even when its panels span
    two thresholds, which is what a reader looking up "Figure B.II" expects.

    ``key`` names the column holding the grouping value; it must already be
    present in each frame. Returns the re-read rows split back out by key, so
    every panel is drawn from the file rather than from the frame that produced
    it.
    """
    order: list[str] = []
    bundles: dict[str, tuple[str, list[pd.DataFrame]]] = {}
    for k, frame in frames.items():
        if key not in frame.columns:
            raise KeyError(f"frame {k!r} has no {key!r} column to group on")
        filename, figure = destinations[k]
        if filename not in bundles:
            bundles[filename] = (figure, [])
            order.append(filename)
        elif bundles[filename][0] != figure:
            raise ValueError(
                f"{filename} is claimed by both {bundles[filename][0]!r} and {figure!r}"
            )
        bundles[filename][1].append(frame)

    published: dict[str, pd.DataFrame] = {}
    for filename in order:
        figure, parts = bundles[filename]
        reread = pd.read_csv(_write(pd.concat(parts, ignore_index=True), filename, figure))
        for k in frames:
            if destinations[k][0] == filename:
                published[k] = reread[reread[key] == k].reset_index(drop=True)
    return published
