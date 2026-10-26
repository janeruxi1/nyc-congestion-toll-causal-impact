"""Check that re-running the notebooks reproduces the committed results.

CI runs every notebook (which rewrites reports/metrics/), then this script compares
each metrics file with the version committed in git. Numbers may differ in the last
digits across machines (BLAS, LightGBM threading), so values are compared with a
relative tolerance instead of byte for byte.

    python scripts/check_metrics.py            # compare with HEAD
"""
from __future__ import annotations

import csv
import io
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
RTOL, ATOL = 1e-4, 1e-6


def committed(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def _num(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def compare(a, b, where: str, problems: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            problems.append(f"{where}: keys differ")
        for k in a.keys() & b.keys():
            compare(a[k], b[k], f"{where}.{k}", problems)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{where}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{where}[{i}]", problems)
    else:
        x, y = _num(a), _num(b)
        if x is not None and y is not None:
            if not (math.isnan(x) and math.isnan(y)) and not math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL):
                problems.append(f"{where}: committed {a} vs new {b}")
        elif a != b:
            problems.append(f"{where}: committed {a!r} vs new {b!r}")


def load(text: str, suffix: str):
    if suffix == ".json":
        return json.loads(text)
    return list(csv.reader(io.StringIO(text)))


def main() -> int:
    problems: list[str] = []
    files = sorted(METRICS.glob("*.json")) + sorted(METRICS.glob("*.csv"))
    for path in files:
        old = load(committed(path), path.suffix)
        new = load(path.read_text(encoding="utf-8"), path.suffix)
        compare(old, new, path.name, problems)
    if problems:
        print("Re-running the notebooks changed the recorded results:")
        print("\n".join(f"  {p}" for p in problems[:50]))
        return 1
    print(f"All {len(files)} metrics files match the committed results (rtol {RTOL}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
