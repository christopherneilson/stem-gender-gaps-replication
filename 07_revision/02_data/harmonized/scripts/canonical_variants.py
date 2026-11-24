#!/usr/bin/env python3
"""Single source for the preferred sample variants and the country ordering.

``create_editor_comment_figures.py`` produces the manuscript figures and holds
the authoritative ``PREFERRED_VARIANT`` and ``COUNTRY_ORDER``. Every other
script here used to keep its own copy, and those copies drifted twice:

* Australia's variant was renamed to ``atar_pooled_2009_2010`` when the ATAR
  tables landed. Scripts still naming ``reply_docx`` matched zero rows and
  carried on, so the setting silently disappeared from their output.
* China was missing from seven maps altogether, so everything they produced
  covered nine settings while reading as ten.

Both failures shrink the result rather than raising, which is why they survived
so long. Parsing the values out of the figure script instead of re-declaring
them removes the failure mode rather than detecting it::

    from canonical_variants import PREFERRED_VARIANT, COUNTRY_ORDER

``COUNTRY_ORDER`` matters independently of the variant map: these scripts cast
``country`` to a ``Categorical`` with it, and a country missing from the list is
dropped even when the variant map is correct.

Not every script should use this. ``create_appendix_d_recent_figures.py``
selects the cohort closest to 2018 for each country, which is a different rule
by design, and orders countries alphabetically; it keeps its own ``SELECTION``
and ``COUNTRY_ORDER``.
"""

from __future__ import annotations

import re
from pathlib import Path

FIGURE_SCRIPT = Path(__file__).with_name("create_editor_comment_figures.py")


def _authority_source() -> str:
    try:
        return FIGURE_SCRIPT.read_text(encoding="utf-8")
    except OSError as exc:                                    # pragma: no cover
        raise SystemExit(f"canonical_variants: cannot read {FIGURE_SCRIPT}: {exc}")


def load_preferred() -> dict[str, str]:
    """Parse PREFERRED_VARIANT out of the figure script."""
    block = re.search(r"^PREFERRED_VARIANT\s*=\s*\{(.*?)\}", _authority_source(), re.S | re.M)
    if not block:
        raise SystemExit(f"canonical_variants: no PREFERRED_VARIANT in {FIGURE_SCRIPT}")
    pairs = re.findall(r"[\"']([^\"']+)[\"']\s*:\s*[\"']([^\"']+)[\"']", block.group(1))
    if not pairs:
        raise SystemExit(f"canonical_variants: PREFERRED_VARIANT in {FIGURE_SCRIPT} is empty")
    return dict(pairs)


def load_country_order() -> list[str]:
    """Parse COUNTRY_ORDER out of the figure script."""
    block = re.search(r"^COUNTRY_ORDER\s*=\s*\[(.*?)\]", _authority_source(), re.S | re.M)
    if not block:
        raise SystemExit(f"canonical_variants: no COUNTRY_ORDER in {FIGURE_SCRIPT}")
    names = re.findall(r"[\"']([^\"']+)[\"']", block.group(1))
    if not names:
        raise SystemExit(f"canonical_variants: COUNTRY_ORDER in {FIGURE_SCRIPT} is empty")
    return names


PREFERRED_VARIANT = load_preferred()
COUNTRY_ORDER = load_country_order()

# The two must describe the same set, or a country selected by the variant map
# is discarded by the Categorical cast (or vice versa) without any error.
_mismatch = set(PREFERRED_VARIANT) ^ set(COUNTRY_ORDER)
if _mismatch:
    raise SystemExit(
        f"canonical_variants: PREFERRED_VARIANT and COUNTRY_ORDER disagree in "
        f"{FIGURE_SCRIPT.name} on {sorted(_mismatch)}. Fix the authority."
    )


if __name__ == "__main__":
    print(f"authority: {FIGURE_SCRIPT}")
    print(f"{len(PREFERRED_VARIANT)} settings")
    for country in COUNTRY_ORDER:
        print(f"  {country:10s} {PREFERRED_VARIANT[country]}")
