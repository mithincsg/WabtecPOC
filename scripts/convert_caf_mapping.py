"""Convert the Change Approval Form workbook to config/caf_mapping.json.

    python scripts/convert_caf_mapping.py

The running app never reads data/CAF.xlsx directly — it reads
config/caf_mapping.json (src/rag/caf_mapping.py), which is small,
dependency-free to parse, and diffable. Re-run this script whenever
data/CAF.xlsx changes; the app picks up the regenerated JSON on its next
request without a restart.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.caf_mapping import read_caf_workbook  # noqa: E402

DEFAULT_SOURCE = REPO_ROOT / "data" / "CAF.xlsx"
DEFAULT_OUTPUT = REPO_ROOT / "config" / "caf_mapping.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="CAF workbook to read.")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Where to write the JSON mapping."
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.is_file():
        print(f"{source} does not exist.", file=sys.stderr)
        return 1

    entries = read_caf_workbook(source)
    try:
        source_label = source.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source_label = source.as_posix()
    payload = {
        "source": source_label,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requirements": {
            entry.requirement_id: {"feature": entry.feature, "section": entry.section}
            for entry in sorted(entries.values(), key=lambda e: e.requirement_id)
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(entries)} requirement mapping(s) from {source} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
