# Replication package: The Global Gender Gap in STEM Applications: Pipeline vs. Choice

Ahimbisibwe, Altmejd, Artemov, Barrios-Fernández, Bizopoulou, Kaila, Liu,
Megalokonomou, Montalbán, Neilson, Otero, Sun and Ye.

This package rebuilds every figure and every results table in the paper and its
online appendix from aggregated counts: for each of the ten admission systems,
the number of women and men in each performance group, and how many of them
ranked a STEM, health or business/law program first, ranked one anywhere, or
were assigned to one. One command reruns the whole pipeline and checks each
output against the published version.

The individual-level records behind the counts are confidential and are not
included. [What is not included](#what-is-not-included) says whom to ask for
each setting.

Version 1.1, built on 2026-11-24 from commit `590c6202967e` of the authors'
working repository.

## Quick start

Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` takes about a minute. It ends with `All reproduced exactly.` and
exit status 0, or lists any output that differs. The versions in
`requirements.txt` are pinned exactly because other versions of pandas and
matplotlib change the files even when every number is the same.

Figure PDFs also need [Ghostscript](https://ghostscript.com/) on the `PATH`
(`gs`, or `gswin64c` on Windows) to embed their fonts. Without it, the SVG
version of every figure is still rebuilt and checked.

## What is reproduced

| Exhibit | Output in `07_revision/02_data/harmonized/figures/` | Panel data in `figures/data/` |
|---|---|---|
| Figure 1 | `fig1_stem_share` | `fig1_revised.csv` |
| Figure 2 | `fig2_pipeline_choice` | `fig2_*.csv` |
| Figure 3 | `fig3_decomposition_scatter` | `fig6.csv` |
| Figure 4 | `fig4_stem_funnel` | `figR0.csv` |
| Figure 5 | `fig5_wage_parity` | `fig4_a.csv`, `fig4_b.csv` |
| Figure B.I | `figB8_thresholds` | `figB8_thresholds.csv` |
| Figure B.II | `figB1_shares_thresholds` | `figB1_*.csv` |
| Figures B.III–B.IV | `figB2_pipeline_levels`, `figB2_pipeline_gaps` | `figB2_*.csv` |
| Figures B.V–B.VI | `figB3_stem_first_levels`, `figB3_stem_first_gaps` | `figB3_*.csv` |
| Figures B.VII–B.VIII | `figB4_stem_any_levels`, `figB4_stem_any_gaps` | `figB4_*.csv` |
| Figure D.I | `figD1_recent_cohort` | `figD1.csv` |
| Table B.I | `table_threshold_robustness.tex` | `threshold_robustness_*.csv` |
| Table B.II | `table_counterfactuals.tex` | from the counts directly |
| Table C.I | `table_crossfield_decomposition.tex`, `vrange_other_fields.tex` | from the counts directly |

Each figure is written as `.svg` and `.pdf`. The remaining figures in the same
folder (`figB5`–`figB7` by score bin, `figC1`–`figC4` for business/law and
health, the `*_thresholds` panels) are the ones the paper defers to its online
materials at <https://gendergapinsights.com>.

Tables A.I and A.III describe the admission systems and are written by hand.
The counts in Table A.II are in
`07_revision/02_data/harmonized/master/top10_source_traceability.csv`.

### The pipeline

`run_all.py` runs these in order:

1. `07_revision/02_data/harmonized/scripts/create_referee_revision_outputs.py`:
   STEM gaps at every threshold, the input to Table B.I. It also writes a few
   exploratory charts that the paper does not use.
2. `figures/build_figure_data.py`: one tidy CSV per figure panel, produced by
   the same functions that select samples and compute rates for the figures.
3. `make_figB8_threshold_panel.py`, in the same scripts folder: the panel for
   Figure B.I.
4. `figures/gallery.py`: draws every figure from those CSVs.
5. `make_threshold_robustness_table.py`, `make_counterfactual_table.py` and
   `make_decomposition_table.py`, in the same scripts folder: Tables B.I, B.II
   and C.I.

The folder layout mirrors the authors' working repository, so the scripts run
unchanged. The one edit is in the three scripts that look for Chile's
student-level file on the authors' machines: that path now points to
`confidential/`, which does not exist here, so the scripts read the cached
aggregates in `cache/`, exactly as they do on a machine without the
microdata.

## Data

### `07_revision/02_data/harmonized/master/harmonized_counts_long.csv`

All counts, one row per setting × sample variant × performance group × outcome
× sex. Columns are defined in `schema/data_dictionary.csv` and outcomes in
`schema/outcome_reference.csv`. Performance groups are deciles of the
setting's admission score, plus the top 5% and top 1% where they can be
formed; groups are always formed within setting and cohort, on the full
distribution of score holders.

Some settings supplied more than one sample. The paper uses one per setting:

| Setting | Variant used | Source agency | Cohorts |
|---|---|---|---|
| Australia (Victoria) | `atar_pooled_2009_2010` | Victorian Tertiary Admissions Centre | 2009–2010 |
| Brazil | `reply_tables` | SISU platform / ENEM records | 2016 |
| Chile | `chile_2012_2018` | DEMRE | 2012–2018 |
| Mainland China (Ningxia) | `china_ningxia_2018` | Ningxia Department of Education | 2018 |
| Finland | `main_pooled` | Ministry of Education (HAREK, AMKOREK) | 2016–2020 |
| Greece | `years_2003_2012` | Ministry of Education | 2003–2012 |
| Spain | `esp_2020_2024_tables` | Ministry of Universities | 2020–2024 |
| Sweden | `reply_docx_2017` | Swedish Council for Higher Education | 2017 |
| Taiwan | `reply_pdf_2010` | Joint Board of College Recruitment Commission | 2010 |
| Uganda | `reply_docx` | Public Universities Joint Admissions Board | 2017/18 |

Other variants (Australia's single-year and mathematics-plus-English rankings,
Finland's 2018 cohort and others) back robustness checks and Appendix D, which
uses the cohort closest to 2018 in each setting. The `notes` column says what
each variant is.

Three caveats the figures carry:

- **Chile.** STEM, health and business/law probabilities are computed among
  students who submitted at least one application; performance groups stay
  defined on all score holders. The applicant-only denominators come from
  student-level data and are cached in `cache/chile_applicant_denominators.csv`
  (and, for Appendix D, `cache/chile_2018_top10_cached.csv`).
- **Greece** reports no top-1% group and no business/law counts at the top 10%
  and top 20%. Those cells are missing, not zero, and the figures mark them.
- **Uganda's** score takes seventeen distinct values, so its performance groups
  cannot be filled exactly; its "top 10%" covers 8.8% of the cohort.

### Other inputs

- `07_revision/02_data/gender_wage_gap_10countries_mixed_sources.csv`: the
  gender wage gaps behind Figure 5's wage-parity axis, with source and URL for
  each setting.
- `05_data/wef_gender_gap_2025_sample.csv`: the World Economic Forum's 2025
  Global Gender Gap index for the ten settings. The Figure 5 script reads it,
  but no exhibit plots it.
- `figure_inputs_top10_summary.csv` in `master/`: the top-10% summary the Figure
  5 script reads.

## What is not included

- **The individual-level administrative records.** They were obtained under
  separate agreements with each agency in the table above and cannot be
  redistributed. Researchers seeking access should apply to that agency.
- **The code that turned those records into the counts.** Each setting's
  research team ran its own extraction on its own data; that code is not part
  of this package.
- **The paper's text and the hand-written tables** (A.I, A.III).

## License and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0, apart from the two third-party series
above (`DATA_LICENSE.md`). If you use either, please cite the paper;
`CITATION.cff` has the reference.

Questions: Christopher Neilson, christopher.neilson@yale.edu.
