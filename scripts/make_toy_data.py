"""Generate the small synthetic dataset shipped under ``examples/toy``.

Run from the repository root::

    python scripts/make_toy_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genepriori.data.toy import build_toy_cache  # noqa: E402


def main() -> None:
    target = ROOT / "examples" / "toy" / "cache"
    build_toy_cache(target)
    print(f"wrote synthetic demo cache to {target}")


if __name__ == "__main__":
    main()
