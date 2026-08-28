"""Two-agent PARTNR runner with a strict-observation PeerConsult layer."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from habitat_llm.evaluation.decentralized_evaluation_runner import (
    DecentralizedEvaluationRunner,
)
from habitat_llm.peer_consult import PeerConsultBoard


class PeerConsultDecentralizedEvaluationRunner(DecentralizedEvaluationRunner):
    """Review high-level proposals before unchanged PARTNR skills execute."""

    def __init__(self, evaluation_runner_config_arg, env_arg) -> None:
        super().__init__(evaluation_runner_config_arg, env_arg)
        conf = evaluation_runner_config_arg.get("peerconsult", {})
        self.board = PeerConsultBoard(
            claim_ttl_decisions=int(conf.get("claim_ttl_decisions", 3)),
            max_targets=int(conf.get("max_targets", 5)),
            max_rooms=int(conf.get("max_rooms", 4)),
            max_reviews=int(conf.get("max_reviews", 2)),
            max_ledger_entries=int(conf.get("max_ledger_entries", 8)),
            max_public_report_chars=int(conf.get("max_public_report_chars", 480)),
            publish_exploration_reports=bool(conf.get("publish_exploration_reports", False)),
            enforce_room_exploration_dedup=bool(conf.get("enforce_room_exploration_dedup", False)),
            enforce_required_recovery=bool(conf.get("enforce_required_recovery", False)),
            enforce_placement_stability=bool(conf.get("enforce_placement_stability", False)),
        )
        self._peer_log_path = os.path.join(self.output_dir, "peer_consult.jsonl")

    def reset_planners(self) -> None:
        super().reset_planners()
        if hasattr(self, "board"):
            self.board.reset()

    @staticmethod
    def _merge(planner_info: Dict[str, Any], one_info: Dict[str, Any]) -> None:
        for key, value in one_info.items():
            if isinstance(value, dict):
                planner_info.setdefault(key, {}).update(value)
            elif isinstance(value, str):
                planner_info[key] = planner_info.get(key, "") + value
            else:
                raise ValueError("Planner logging values must be dictionaries or strings")

    def _write_event(self, event: Dict[str, Any]) -> None:
        with open(self._peer_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def get_low_level_actions(
        self, instruction: str, observations: Dict[str, Any], world_graph: Dict[int, Any]
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        # Deliberately use only `world_graph`, never `full_world_graph` or metrics.
        self.board.observe(world_graph)
        cards = {uid: self.board.decision_card(uid) for uid in self.planner}
        proposals: Dict[int, Dict[str, Any]] = {}
        assert isinstance(self.planner, dict)
        for uid, planner in sorted(self.planner.items()):
            if not hasattr(planner, "set_decision_card"):
                raise TypeError("PeerConsult runner requires PeerConsultZeroShotReactPlanner")
            planner.set_decision_card(cards[uid])
            proposals[uid] = planner.prepare_proposal(instruction, observations, world_graph)

        final_actions, reviews, intents = self.board.review(proposals)
        low_level_actions: Dict[int, Any] = {}
        planner_info: Dict[str, Any] = {}
        all_done = True
        for uid, planner in sorted(self.planner.items()):
            actions, info, is_done = planner.execute_proposal(
                proposals[uid], final_actions[uid], observations,
                review=next((review for review in reviews if review["agent"] == uid), None),
                intent=intents[uid],
            )
            low_level_actions.update(actions)
            self._merge(planner_info, info)
            all_done = all_done and is_done

        self.board.record_execution_evidence(planner_info)
        # Let the generic runner count planning decisions against V4's own
        # monotonic public task facts rather than incidental graph changes.
        planner_info["peerconsult_progress_signature"] = self.board.progress_signature()
        event = self.board.event(cards, proposals, reviews, final_actions, intents, planner_info)
        # The JSONL file spans every episode in a val_mini run.  Attach the
        # public episode filename so PeerConsult decisions can be joined with
        # the official per-episode prompt, planner-log, and detailed-trace
        # artifacts during failure analysis.
        event["episode_filename"] = self.episode_filename
        self._write_event(event)
        return low_level_actions, planner_info, all_done
