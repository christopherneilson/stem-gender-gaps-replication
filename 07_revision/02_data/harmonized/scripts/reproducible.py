#!/usr/bin/env python3
"""Make regenerated figures byte-identical when the underlying data has not moved.

matplotlib stamps a ``CreationDate`` into every PDF it writes, taken from the
clock. Re-running a figure script therefore rewrites the file even when every
number in it is unchanged, and since this repo commits its figures, that shows
up as a modified file in ``git status`` and as a binary diff in review. The
noise is indistinguishable from a real change until someone opens both copies.

matplotlib honours the ``SOURCE_DATE_EPOCH`` convention: when it is set, the
timestamp is taken from it instead of the clock. Importing this module sets it,
so a rerun that changes no data produces no diff::

    import reproducible  # noqa: F401  (import for side effect)

Verified: without it, two consecutive runs of make_stem_funnel.py produce
different PDF bytes; with it they are identical. PNG output was already
deterministic, so only the PDF backend is affected.

``setdefault`` is deliberate. A build system or CI job that sets its own
``SOURCE_DATE_EPOCH`` keeps it; this module only supplies a value when nothing
else has.
"""

from __future__ import annotations

import os

# 2026-01-01T00:00:00Z. The exact instant carries no meaning and any fixed value
# would do, but 0 renders as a 1969 date in local time, which reads as a bug.
SOURCE_DATE_EPOCH = "1767225600"

os.environ.setdefault("SOURCE_DATE_EPOCH", SOURCE_DATE_EPOCH)


if __name__ == "__main__":
    import datetime as _dt

    stamp = int(os.environ["SOURCE_DATE_EPOCH"])
    print(f"SOURCE_DATE_EPOCH={stamp} "
          f"({_dt.datetime.fromtimestamp(stamp, _dt.timezone.utc).isoformat()})")
