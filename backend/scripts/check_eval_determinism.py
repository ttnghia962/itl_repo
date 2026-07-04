"""Verify run_evaluation.py produces identical output across fresh process
launches (Phase 0 done-criterion): same retrieved IDs per query, in order,
and metric values matching to 4 decimal places."""
import json
import subprocess
import sys

N = 6


def run_once() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "app.evaluation.run_evaluation"],
        capture_output=True,
        text=True,
        check=True,
        cwd="backend",
    )
    return json.loads(proc.stdout)


def round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj


def main() -> None:
    results = [round_floats(run_once()) for _ in range(N)]
    baseline = results[0]
    all_match = all(r == baseline for r in results[1:])
    print(f"{N}/{N} runs identical: {all_match}")
    if not all_match:
        for i, r in enumerate(results):
            print(f"--- run {i} ---")
            print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
