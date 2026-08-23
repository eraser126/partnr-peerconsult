#!/usr/bin/env python3
"""Create an auditable five-episode subset from the original val_mini split."""

import gzip
import json
from pathlib import Path

ROOT = Path("/data/user/hd68631/projects/partnr-planner")
SOURCE = ROOT / "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
OUTPUT = ROOT / "data/datasets/partnr_episodes/v0_0/val_mini_repair_5.json.gz"
FAILED_IDS = {"209", "218", "271", "288", "357"}

with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
    dataset = json.load(handle)

dataset["episodes"] = [
    episode for episode in dataset["episodes"] if str(episode["episode_id"]) in FAILED_IDS
]
actual_ids = {str(episode["episode_id"]) for episode in dataset["episodes"]}
if actual_ids != FAILED_IDS:
    raise RuntimeError(f"Unexpected repair subset IDs: {sorted(actual_ids)}")

with gzip.open(OUTPUT, "wt", encoding="utf-8") as handle:
    json.dump(dataset, handle)

print(f"Wrote {OUTPUT} with {len(dataset['episodes'])} episodes: {sorted(actual_ids)}")
