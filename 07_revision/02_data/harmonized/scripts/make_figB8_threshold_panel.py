#!/usr/bin/env python3
r"""Build the panel behind Appendix Figure B.VIII, the choice gap at the top 1%,
5% and 10%.

The editor asked what happens at the very top of the distribution. Table B.I
answers the broadening question (top 5% out to top 30%); this panel answers the
narrowing one, and it is the only exhibit reporting the top 1%.

Denominators are all students in the score group, for every setting and every
threshold, which is the convention of `threshold_robustness_by_country.csv` and
therefore of Table B.I. That is a deliberate departure from Figures B.I-B.IV,
where Chile conditions on applicants: `cache/chile_applicant_denominators.csv`
has no top-1% row, so the applicant restriction cannot be applied at the top 1%
and mixing the two conventions inside one figure would put Chile's three bars on
three different denominators. Reading the whole figure on score holders keeps it
internally consistent and makes it reconcile exactly with the table beside it.
The caption says so.

Greece has no top-1% row -- its data arrive by decile and top 5% -- so it shows
two bars rather than three. `bar_panel` in figures/gallery.py handles a missing
series by drawing nothing for it.

Reads  02_data/harmonized/figures/threshold_robustness_by_country.csv
Writes figures/data/figB8_thresholds.csv

Run:  python 07_revision/02_data/harmonized/scripts/make_figB8_threshold_panel.py
"""

from __future__ import annotations

import csv
from pathlib import Path

HARMONIZED = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
SRC = HARMONIZED / "figures" / "threshold_robustness_by_country.csv"
OUT = ROOT / "figures" / "data" / "figB8_thresholds.csv"

OUTCOME = "stem_first"
# Series label per threshold, in the order they should be drawn within a country.
SERIES = [(1, "Top 1%"), (5, "Top 5%"), (10, "Top 10%")]

# The one order every exhibit uses (editor comment 8); mirrors ORDER in
# figures/gallery.py, which re-sorts to it anyway.
ORDER = ["Taiwan", "Chile", "Brazil", "China", "Uganda", "Finland", "Spain",
         "Australia", "Greece", "Sweden"]

FIELDS = ["x", "series", "y", "ci_lo", "ci_hi", "n_obs",
          "threshold", "outcome", "denominator"]


def main() -> int:
    with SRC.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["outcome"] == OUTCOME]

    by_key = {(r["country"], int(r["threshold"])): r for r in rows}

    out, missing = [], []
    for country in ORDER:
        for thr, label in SERIES:
            r = by_key.get((country, thr))
            # An absent row and a present-but-empty gap both mean the group
            # cannot be formed; Greece at the top 1% is the latter.
            if r is None or not r["gap_female_minus_male"]:
                missing.append(f"{country} {label}")
                continue
            n = int(float(r["pipeline_male"]) + float(r["pipeline_female"]))
            out.append({
                "x": country,
                "series": label,
                "y": 100 * float(r["gap_female_minus_male"]),
                "ci_lo": "", "ci_hi": "",
                "n_obs": n,
                "threshold": label,
                "outcome": OUTCOME,
                "denominator": "all in score group",
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    print(f"wrote {len(out)} rows to {OUT.relative_to(ROOT)}")
    # Never silent about a dropped cell: a missing bar is a claim about the data.
    print("omitted (group cannot be formed):",
          ", ".join(missing) if missing else "none")
    vals = [r["y"] for r in out if r["series"] == "Top 1%"]
    print(f"top 1%: {len(vals)} settings, all negative = {all(v < 0 for v in vals)}, "
          f"mean {sum(vals) / len(vals):.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
