#!/usr/bin/env python3
"""Merge original and repaired PARTNR Heuristic-Expert val_mini results."""

import csv
import gzip
import json
from pathlib import Path


ROOT = Path("/data/user/hd68631/projects/partnr-planner")
ORIGINAL = ROOT / "logs/heuristic-val-mini-336811/results/episode_result_log.csv"
REPAIRED = ROOT / "logs/heuristic-val-mini-repair-336814/results/episode_result_log.csv"
DATASET = ROOT / "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
OUTPUT = ROOT / "logs/heuristic-val-mini-complete-336811-336814/results"
NUMERIC_FIELDS = [
    "runtime",
    "sim_step_count",
    "task_percent_complete",
    "task_state_success",
]


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


original = read_rows(ORIGINAL)
repaired = read_rows(REPAIRED)
rows_by_id = {row["episode_id"]: row for row in original}
if len(rows_by_id) != len(original):
    raise RuntimeError("Duplicate episode IDs in the original run")

for row in repaired:
    episode_id = row["episode_id"]
    if episode_id in rows_by_id:
        raise RuntimeError(f"Repair run overlaps original result: {episode_id}")
    rows_by_id[episode_id] = row

with gzip.open(DATASET, "rt", encoding="utf-8") as handle:
    dataset_ids = [str(episode["episode_id"]) for episode in json.load(handle)["episodes"]]

if set(rows_by_id) != set(dataset_ids):
    missing = sorted(set(dataset_ids) - set(rows_by_id))
    unexpected = sorted(set(rows_by_id) - set(dataset_ids))
    raise RuntimeError(f"Coverage mismatch; missing={missing}, unexpected={unexpected}")

merged = [rows_by_id[episode_id] for episode_id in dataset_ids]
averages = {
    field: sum(float(row[field]) for row in merged) / len(merged)
    for field in NUMERIC_FIELDS
}
success_count = sum(int(float(row["task_state_success"])) for row in merged)
failure_count = len(merged) - success_count

OUTPUT.mkdir(parents=True, exist_ok=True)
with (OUTPUT / "episode_result_log.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=merged[0].keys())
    writer.writeheader()
    writer.writerows(merged)

for name in ("run_result_log.csv", "end_result_log.csv"):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NUMERIC_FIELDS)
        writer.writeheader()
        writer.writerow(averages)

summary = {
    "benchmark": "PARTNR val_mini",
    "baseline": "Heuristic-Expert / ScriptedCentralizedPlanner",
    "episodes": len(merged),
    "task_successes": success_count,
    "task_failures": failure_count,
    "metrics": averages,
    "source_jobs": {
        "original": {"job_id": 336811, "episodes": len(original)},
        "repair": {"job_id": 336814, "episodes": len(repaired)},
    },
    "repair_scope": [209, 218, 271, 288, 357],
    "repair_notes": [
        "world-graph serialization records parentless objects as unknown instead of aborting logging",
        "state predicates retain candidates used by earlier terminal rearrangement propositions",
    ],
}
with (OUTPUT / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")

print(json.dumps(summary, indent=2))
