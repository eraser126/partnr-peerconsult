#!/usr/bin/env python3
"""Build a deterministic 56-episode representative PARTNR val_mini subset.

The subset is stratified by PARTNR's dataset-provided ``task_gen`` category,
goal predicate families, task size, temporal dependencies, and argument
constraints.  It is an evaluation subset, not a training split.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DATASET = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
QUOTAS = {"rearrange": 18, "temporal": 18, "spatial": 12, "object_states": 8}
SEED = "peerconsult-v4-partnr-representative-v1"


def stable_key(episode: Dict[str, Any]) -> str:
    value = "{}:{}".format(SEED, episode["episode_id"])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def complexity(count: int) -> str:
    if count <= 2:
        return "short_1_2"
    if count <= 4:
        return "medium_3_4"
    if count <= 7:
        return "long_5_7"
    return "very_long_8_plus"


def temporal_edge_bucket(episode: Dict[str, Any]) -> str:
    edge_count = sum(
        len(constraint.get("args", {}).get("dag_edges", []))
        for constraint in episode.get("evaluation_constraints", [])
        if constraint.get("type") == "TemporalConstraint"
    )
    return "edges_0" if edge_count == 0 else "edges_1" if edge_count == 1 else "edges_2_plus"


def episode_features(episode: Dict[str, Any]) -> Tuple[Tuple[str, ...], str, str, str]:
    predicates = tuple(sorted({p.get("function_name", "unknown") for p in episode["evaluation_propositions"]}))
    size = complexity(len(episode["evaluation_propositions"]))
    edge_bucket = temporal_edge_bucket(episode)
    constraints = tuple(
        sorted(
            constraint.get("type", "unknown")
            for constraint in episode.get("evaluation_constraints", [])
            if constraint.get("type") not in {"TemporalConstraint", "TerminalSatisfactionConstraint"}
        )
    ) or ("none",)
    features = {
        "size:" + size,
        "temporal_edges:" + edge_bucket,
        *["predicate:" + predicate for predicate in predicates],
        *["constraint:" + constraint for constraint in constraints],
    }
    return tuple(sorted(features)), size, edge_bucket, "+".join(predicates)


def choose_group(candidates: List[Dict[str, Any]], quota: int) -> List[Dict[str, Any]]:
    """Greedily cover rare structural features, then round-robin strata."""
    feature_frequency = Counter(feature for episode in candidates for feature in episode["features"])
    uncovered = set(feature_frequency)
    remaining = sorted(candidates, key=lambda episode: stable_key(episode["raw"]))
    selected: List[Dict[str, Any]] = []
    while remaining and len(selected) < quota and uncovered:
        def score(episode: Dict[str, Any]) -> Tuple[float, str]:
            coverage = sum(1.0 / feature_frequency[feature] for feature in episode["features"] if feature in uncovered)
            return coverage, stable_key(episode["raw"])

        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
        uncovered.difference_update(best["features"])

    by_stratum: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for episode in remaining:
        by_stratum[(episode["complexity"], episode["predicate_signature"], episode["temporal_edges"])].append(episode)
    for episodes in by_stratum.values():
        episodes.sort(key=lambda episode: stable_key(episode["raw"]))
    strata = sorted(by_stratum)
    while len(selected) < quota:
        progressed = False
        for stratum in strata:
            if by_stratum[stratum] and len(selected) < quota:
                selected.append(by_stratum[stratum].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != quota:
        raise RuntimeError("Could select {} episodes from a quota of {}".format(len(selected), quota))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument(
        "--output", default="scripts/subsets/partnr_representative_v1.json"
    )
    args = parser.parse_args()
    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(episodes):
        category = raw.get("info", {}).get("task_gen")
        if category not in QUOTAS:
            raise RuntimeError("Unsupported or missing PARTNR task_gen: {}".format(category))
        features, size, edges, predicates = episode_features(raw)
        grouped[category].append(
            {
                "index": index,
                "raw": raw,
                "features": features,
                "complexity": size,
                "temporal_edges": edges,
                "predicate_signature": predicates,
            }
        )

    selected = []
    for category, quota in QUOTAS.items():
        if len(grouped[category]) < quota:
            raise RuntimeError("{} has only {} candidates for quota {}".format(category, len(grouped[category]), quota))
        selected.extend(choose_group(grouped[category], quota))
    selected.sort(key=lambda episode: episode["index"])

    records = []
    for episode in selected:
        raw = episode["raw"]
        records.append(
            {
                "episode_index": episode["index"],
                "episode_id": str(raw["episode_id"]),
                "task_gen": raw["info"]["task_gen"],
                "complexity": episode["complexity"],
                "temporal_edges": episode["temporal_edges"],
                "predicate_signature": episode["predicate_signature"],
                "features": episode["features"],
                "instruction": raw["instruction"],
            }
        )
    manifest = {
        "name": "PARTNR val_mini representative subset v1",
        "purpose": "56-episode stratified evaluation subset for PeerConsult V4",
        "source_dataset": args.dataset,
        "seed": SEED,
        "task_quotas": QUOTAS,
        "episode_indices": [record["episode_index"] for record in records],
        "category_counts": dict(Counter(record["task_gen"] for record in records)),
        "episodes": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "episodes": len(records), "category_counts": manifest["category_counts"]}, indent=2))


if __name__ == "__main__":
    main()
