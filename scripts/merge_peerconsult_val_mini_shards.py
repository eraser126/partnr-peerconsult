#!/usr/bin/env python3
"""Merge disjoint PARTNR val_mini array shards and validate full coverage."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List


METRICS = [
    "runtime",
    "sim_step_count",
    "task_percent_complete",
    "task_state_success",
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_glob", help="Glob for array run directories, e.g. logs/peerconsult-tatu-val-mini-12345_*")
    parser.add_argument(
        "--dataset",
        default="data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
        help="PARTNR episode split used by every array task",
    )
    parser.add_argument(
        "--method",
        default="PeerConsult V2 / OpenAI-compatible API",
        help="Method label written to the merged summary",
    )
    args = parser.parse_args()

    run_dirs = sorted(Path().glob(args.run_glob))
    if not run_dirs:
        raise SystemExit(f"No run directories match: {args.run_glob}")

    rows_by_id: Dict[str, Dict[str, str]] = {}
    source_shards = []
    for run_dir in run_dirs:
        csv_path = run_dir / "results" / "episode_result_log.csv"
        if not csv_path.is_file():
            raise SystemExit(f"Missing shard result: {csv_path}")
        rows = read_rows(csv_path)
        if not rows:
            raise SystemExit(f"Empty shard result: {csv_path}")
        for row in rows:
            episode_id = row["episode_id"]
            if episode_id in rows_by_id:
                raise SystemExit(f"Duplicate episode {episode_id} in {csv_path}")
            rows_by_id[episode_id] = row
        source_shards.append({"directory": str(run_dir), "episodes": len(rows)})

    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        dataset_ids = [str(x["episode_id"]) for x in json.load(handle)["episodes"]]
    expected, actual = set(dataset_ids), set(rows_by_id)
    if actual != expected:
        raise SystemExit(
            "Coverage mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )

    merged = [rows_by_id[episode_id] for episode_id in dataset_ids]
    averages = {
        metric: sum(float(row[metric]) for row in merged) / len(merged)
        for metric in METRICS
    }
    successes = sum(int(float(row["task_state_success"])) for row in merged)

    run_prefix = run_dirs[0].name.rsplit("_", 1)[0]
    output = run_dirs[0].parent / f"{run_prefix}_merged" / "results"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episode_result_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged[0].keys())
        writer.writeheader()
        writer.writerows(merged)
    for name in ("run_result_log.csv", "end_result_log.csv"):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRICS)
            writer.writeheader()
            writer.writerow(averages)

    summary = {
        "benchmark": "PARTNR val_mini",
        "method": args.method,
        "episodes": len(merged),
        "task_successes": successes,
        "task_failures": len(merged) - successes,
        "metrics": averages,
        "source_shards": source_shards,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
