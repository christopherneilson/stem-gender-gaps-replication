#!/usr/bin/env python3
"""Rebuild every figure and table in the paper from the aggregated counts, and
check each one against the published version shipped in this package.

Run from the package root, in an environment built from requirements.txt:

    python run_all.py

The script records the published outputs first, reruns the pipeline over them,
and compares. It exits 0 only if every output is reproduced. Comparison ignores
line endings, so a Windows checkout that converted them still passes.

Figure PDFs need Ghostscript on the PATH (`gs`, or `gswin64c` on Windows) to
embed their fonts. Without it the SVG versions are still rebuilt and checked,
and the PDFs shipped here are left as they are.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HARMONIZED = ROOT / "07_revision" / "02_data" / "harmonized"
SCRIPTS = HARMONIZED / "scripts"
FIGURES = HARMONIZED / "figures"

# In order: each step reads what the ones before it wrote.
STEPS = [
    ("Threshold-robustness counts (input to Table B.I)", SCRIPTS / "create_referee_revision_outputs.py"),
    ("Figure panel data", ROOT / "figures" / "build_figure_data.py"),
    ("Figure B.I panel data", SCRIPTS / "make_figB8_threshold_panel.py"),
    ("Figures (SVG, and PDF when Ghostscript is available)", ROOT / "figures" / "gallery.py"),
    ("Table B.I", SCRIPTS / "make_threshold_robustness_table.py"),
    ("Table B.II", SCRIPTS / "make_counterfactual_table.py"),
    ("Table C.I", SCRIPTS / "make_decomposition_table.py"),
    ("Wage-parity sensitivity and prose", SCRIPTS / "build_wage_parity_sensitivity.py"),
]

# Everything the pipeline writes that the paper or its online materials print.
CHECKED = (
    sorted((ROOT / "figures" / "data").glob("*.csv"))
    + sorted(FIGURES.glob("*.svg"))
    + [FIGURES / n for n in (
        "threshold_robustness_by_country.csv",
        "threshold_robustness_pooled.csv",
        "table_threshold_robustness.tex",
        "table_counterfactuals.tex",
        "table_crossfield_decomposition.tex",
        "vrange_other_fields.tex",
        "wage_parity_sensitivity.json",
        "wage_parity_sensitivity.tex",
        "wage_parity_main_text.tex",
    )]
)


def digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    missing = [p for p in CHECKED if not p.exists()]
    if missing:
        print("Published outputs missing from the package; cannot compare:")
        for p in missing:
            print("  ", p.relative_to(ROOT))
        return 1
    before = {p: digest(p) for p in CHECKED}
    stamp = {p: p.stat().st_mtime_ns for p in CHECKED}
    ghostscript = shutil.which("gs") or shutil.which("gswin64c")

    for label, script in STEPS:
        print(f"\n== {label}: {script.relative_to(ROOT)}")
        run = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                             capture_output=True, text=True)
        if run.returncode != 0:
            if script.name == "gallery.py" and not ghostscript:
                print("   PDF export skipped: Ghostscript not found. SVGs are checked below.")
                continue
            print(run.stdout[-2000:])
            print(run.stderr[-2000:])
            print(f"   FAILED (exit {run.returncode})")
            return 1
        print("   ok")

    stale = [p for p in CHECKED if p.stat().st_mtime_ns == stamp[p]]
    if stale:
        print("\nThese outputs were not rewritten by the pipeline, so they were not checked:")
        for p in stale:
            print("  ", p.relative_to(ROOT))
        return 1
    changed = [p for p in CHECKED if digest(p) != before[p]]
    print(f"\nChecked {len(CHECKED)} published outputs against the rebuild.")
    if changed:
        print(f"{len(changed)} differ:")
        for p in changed:
            print("  ", p.relative_to(ROOT))
        return 1
    print("All reproduced exactly.")
    if not ghostscript:
        print("(Figure PDFs were not rebuilt: install Ghostscript to rebuild them too.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
