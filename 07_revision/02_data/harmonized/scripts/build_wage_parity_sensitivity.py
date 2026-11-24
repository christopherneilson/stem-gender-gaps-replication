"""Recompute the wage-parity sensitivity statements from counts and wage sources.

Run from the repository root before building the manuscript or site pack.
Use --check to detect stale generated prose or statistics without writing them.
OLS is unweighted across settings, with an intercept and conventional residual
standard errors (n - 2 degrees of freedom). These are descriptive correlations.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import make_decomposition_table as decomposition

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "07_revision/02_data/harmonized/figures"

# Decimal places kept in the JSON record. The least-squares solve's last binary
# digit depends on the numerical library, so at full precision the file differed
# between machines (-2.053131269050448 here, -2.0531312690504473 there) even with
# every reported number identical, and every byte-for-byte check of it failed
# off the machine that wrote it: verify_claims.py, this script's --check (which
# build_site_inputs.py runs), and the replication package's run_all.py. Six
# places is three more than anything printed and leaves the file the same
# wherever it is regenerated.
JSON_DECIMALS = 6


def _rounded(obj):
    if isinstance(obj, float):
        return round(float(obj), JSON_DECIMALS)
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v) for v in obj]
    return obj


def estimates():
    wage = pd.read_csv(ROOT / "07_revision/02_data/gender_wage_gap_10countries_mixed_sources.csv")
    wage = wage.set_index("country").gender_wage_gap_percent
    counts = decomposition.load()
    cache = pd.read_csv(ROOT / "07_revision/02_data/harmonized/cache/chile_applicant_denominators.csv", comment="#").set_index("score_group").loc["Decile 10"]
    rows = []
    for country in sorted(decomposition.PREFERRED):
        tm, tf = [counts[country, "pipeline_total", s] for s in ("male", "female")]
        am, af = [counts[country, "stem_first", s] for s in ("male", "female")]
        dm, df = (cache["male"], cache["female"]) if country == "Chile" else (tm, tf)
        rows.append(dict(country=country, wage_parity=1-wage[country]/100,
                         pipeline_log=math.log(tm/tf), choice_log=math.log((am/tm)/(af/tf)),
                         pipeline_pp=100*(tf-tm)/(tf+tm), choice_pp=100*(af/df-am/dm)))

    def fit(selected, outcome):
        x = np.array([[1, r["wage_parity"]] for r in selected])
        y = np.array([r[outcome] for r in selected])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual_variance = float(((y-x@beta)**2).sum()/(len(y)-2))
        se = math.sqrt(residual_variance*np.linalg.inv(x.T@x)[1, 1])
        return dict(n=len(y), slope=float(beta[1]), standard_error=se, t=float(beta[1]/se))

    result = {"method": "Unweighted OLS with intercept; conventional standard errors; residual df = n - 2",
              "inputs": rows, "estimates": {}}
    for outcome in ("pipeline_log", "choice_log", "pipeline_pp", "choice_pp"):
        result["estimates"][outcome] = {
            "all": fit(rows, outcome),
            "leave_one_out": {r["country"]: fit([s for s in rows if s["country"] != r["country"]], outcome) for r in rows},
        }
    return result


def outputs():
    stats = estimates()
    e = stats["estimates"]
    p, c = e["pipeline_log"], e["choice_log"]
    pp, cp = e["pipeline_pp"]["all"], e["choice_pp"]["all"]
    loo = [v["slope"] for v in c["leave_one_out"].values()]
    text = (
        f"In log-ratio units, the unweighted slopes per unit of wage parity are ${p['all']['slope']:.2f}$ "
        f"for $P$ ($t={p['all']['t']:.1f}$) and ${c['all']['slope']:+.2f}$ for $C$ ($t={c['all']['t']:+.1f}$). "
        r"These components use full top-decile totals, including Chile, as in Table \ref{tab:crossfield_decomposition}. "
        r"Figure \ref{fig:gender_norms} instead plots female-minus-male percentage-point gaps and conditions Chile's application rates on submitted lists. "
        f"Pipeline precision is similar on either scale ($|t|={abs(p['all']['t']):.2f}$ in log points versus ${abs(pp['t']):.2f}$ in percentage points). "
        f"Leave-one-setting-out slopes on $C$ range from ${min(loo):+.2f}$ to ${max(loo):+.2f}$. "
        f"Excluding Uganda changes the slope on $C$ to ${c['leave_one_out']['Uganda']['slope']:+.2f}$ "
        f"($t={c['leave_one_out']['Uganda']['t']:+.1f}$); both the log-ratio and percentage-point choice slopes change sign. "
        "The persistent choice gap is robust across settings; the direction and precision of its association with wage parity are sensitive to scale and sample composition.\n"
    )
    main = (
        r"Figure \ref{fig:gender_norms} relates both gaps to wage parity, defined as one minus the reported gender wage gap, "
        "with both panels on the same female-minus-male percentage-point scale. Women's representation among high achievers "
        f"rises with wage parity (panel a): a 0.1 increase is associated with about {round(pp['slope']/10)} percentage points "
        "in the pipeline gap in women's favor. The choice gap is comparatively flat (panel b): the same increase is associated "
        f"with about {abs(cp['slope']/10):.1f} percentage points in men's favor, a slope not statistically different from zero "
        "that reverses sign when Uganda is excluded."
        r"\footnote{Online \ref{sec:appendix_wage_parity} describes the differing wage concepts, coverage and years and reports sensitivity checks on both scales.} "
        "These are descriptive correlations across ten settings, not effects of wage parity on either gap.\n"
    )
    # The prose above formats full-precision values; only the record is rounded.
    return {OUT / "wage_parity_sensitivity.json": json.dumps(_rounded(stats), indent=2)+"\n",
            OUT / "wage_parity_sensitivity.tex": text,
            OUT / "wage_parity_main_text.tex": main}


def main():
    check = argparse.ArgumentParser()
    check.add_argument("--check", action="store_true")
    args = check.parse_args()
    stale = []
    for path, text in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            # LF on every platform, like the other generators: on Windows the
            # default newline would rewrite these files with CRLF, which breaks
            # the replication package's MANIFEST.sha256 even though run_all.py,
            # comparing modulo line endings, still reports them identical.
            path.write_text(text, encoding="utf-8", newline="\n")
    if stale:
        raise SystemExit("Stale wage-parity outputs: " + ", ".join(stale))
    print("PASS: wage-parity estimates and generated prose agree with source data" if args.check else "Generated wage-parity sensitivity estimates and prose")


if __name__ == "__main__":
    main()
