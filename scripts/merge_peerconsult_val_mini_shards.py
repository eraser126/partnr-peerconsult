#!/usr/bin/env python3
"""Merge disjoint PARTNR val_mini shards and validate their intended coverage."""

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
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Merge only the first N dataset episodes; 0 requires full split coverage.",
    )
    parser.add_argument(
        "--episode-indices-file",
        type=Path,
        help="JSON manifest containing an `episode_indices` list. Overrides --max-episodes.",
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
        all_dataset_ids = [str(x["episode_id"]) for x in json.load(handle)["episodes"]]
    if args.max_episodes < 0:
        raise SystemExit("--max-episodes must be non-negative")
    if args.episode_indices_file:
        if not args.episode_indices_file.is_file():
            raise SystemExit(f"Missing episode manifest: {args.episode_indices_file}")
        manifest = json.loads(args.episode_indices_file.read_text(encoding="utf-8"))
        indices = manifest.get("episode_indices")
        if not isinstance(indices, list) or not indices:
            raise SystemExit("Episode manifest must contain a non-empty `episode_indices` list")
        if any(not isinstance(index, int) for index in indices) or len(set(indices)) != len(indices):
            raise SystemExit("Episode manifest indices must be unique integers")
        if any(index < 0 or index >= len(all_dataset_ids) for index in indices):
            raise SystemExit("Episode manifest contains an out-of-range index")
        dataset_ids = [all_dataset_ids[index] for index in indices]
    else:
        dataset_ids = all_dataset_ids[: args.max_episodes] if args.max_episodes else all_dataset_ids
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
    if args.episode_indices_file:
        summary["episode_manifest"] = str(args.episode_indices_file)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
