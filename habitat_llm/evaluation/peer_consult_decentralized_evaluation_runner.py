"""Two-agent PARTNR runner with a strict-observation PeerConsult layer."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Tuple

from habitat_llm.evaluation.decentralized_evaluation_runner import (
    DecentralizedEvaluationRunner,
)
from habitat_llm.peer_consult import PeerConsultBoard


class PeerConsultDecentralizedEvaluationRunner(DecentralizedEvaluationRunner):
    """Review high-level proposals before unchanged PARTNR skills execute."""

    _COMPLETION_MEASURES = (
        "auto_eval_proposition_tracker",
        "task_constraint_validation",
        "task_percent_complete",
        "task_state_success",
        "task_evaluation_log",
        "task_explanation",
    )

    def __init__(self, evaluation_runner_config_arg, env_arg) -> None:
        super().__init__(evaluation_runner_config_arg, env_arg)
        conf = evaluation_runner_config_arg.get("peerconsult", {})
        self.board = PeerConsultBoard(
            claim_ttl_decisions=int(conf.get("claim_ttl_decisions", 3)),
            max_targets=int(conf.get("max_targets", 5)),
            max_rooms=int(conf.get("max_rooms", 4)),
            max_reviews=int(conf.get("max_reviews", 2)),
            max_execution_facts=int(conf.get("max_execution_facts", 6)),
            max_ledger_entries=int(conf.get("max_ledger_entries", 8)),
            max_public_report_chars=int(conf.get("max_public_report_chars", 480)),
            publish_exploration_reports=bool(conf.get("publish_exploration_reports", False)),
            enforce_room_exploration_dedup=bool(conf.get("enforce_room_exploration_dedup", False)),
            enforce_required_recovery=bool(conf.get("enforce_required_recovery", False)),
            enforce_placement_stability=bool(conf.get("enforce_placement_stability", False)),
            share_discovered_locations=bool(conf.get("share_discovered_locations", False)),
            max_shared_locations=int(conf.get("max_shared_locations", 5)),
        )
        # Composite skills can remain ``ongoing`` indefinitely when a local
        # motion controller is blocked.  This is a runner-side safety bound,
        # independent of the LLM's planning budget.
        self.max_action_wall_time_s = float(conf.get("max_action_wall_time_s", 300))
        self._active_action_started_at: Dict[Tuple[int, int], float] = {}
        self._peer_log_path = os.path.join(self.output_dir, "peer_consult.jsonl")

    def reset_planners(self) -> None:
        super().reset_planners()
        if hasattr(self, "board"):
            self.board.reset()
        if hasattr(self, "_active_action_started_at"):
            self._active_action_started_at.clear()

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

    def _abort_stalled_actions(
        self, planner_info: Dict[str, Any], low_level_actions: Dict[int, Any]
    ) -> None:
        """End only a skill that remains non-terminal past a wall-clock bound.

        A terminal failure ticket is fed back through the usual board evidence
        path.  Consequently the claim is released and no public placement is
        recorded; the planner's next prompt sees a factual failure instead of
        silently treating an abandoned proposal as a completed transport.
        """
        if self.max_action_wall_time_s <= 0:
            return
        tickets = planner_info.get("peerconsult_action_ticket", {})
        now = time.monotonic()
        for uid_raw, ticket in list(tickets.items()):
            uid = int(uid_raw)
            ticket_id = ticket.get("id")
            if ticket_id is None:
                continue
            key = (uid, int(ticket_id))
            if ticket.get("terminal"):
                self._active_action_started_at.pop(key, None)
                continue
            started_at = self._active_action_started_at.setdefault(key, now)
            if now - started_at < self.max_action_wall_time_s:
                continue
            reason = "watchdog: action exceeded {:.0f}s without terminal executor response".format(
                self.max_action_wall_time_s
            )
            planner = self.planner[uid]
            planner.abort_active_action(reason)
            low_level_actions.pop(uid, None)
            aborted_ticket = dict(ticket)
            aborted_ticket.update(
                status="terminal",
                terminal=True,
                outcome="terminal_failure",
                response=reason,
            )
            tickets[uid_raw] = aborted_ticket
            planner_info.setdefault("responses", {})[uid] = reason
            planner_info.setdefault("replan_required", {})[uid] = True
            planner_info.setdefault("replanned", {})[uid] = False
            planner_info.setdefault("high_level_actions", {}).pop(uid, None)
            planner_info.setdefault("peerconsult_execution_actions", {}).pop(uid, None)
            self._active_action_started_at.pop(key, None)

    def _official_completion_reached(self) -> bool:
        """Query only PARTNR's official success bit, never its goal explanation.

        The base evaluation loop otherwise ends only when a planner emits
        Done[].  V4 deliberately keeps that decision out of the LLM, so this
        adapter provides the required runner-side completion boundary.
        """
        try:
            curr_env = self.env_interface.env.env.env._env
            measures = curr_env.task.measurements.measures
            for measure_name in self._COMPLETION_MEASURES:
                if measure_name in measures:
                    measures[measure_name].update_metric(
                        task=curr_env.task, episode=curr_env.current_episode
                    )
            success = measures.get("task_state_success")
            return bool(success and success.get_metric())
        except (AttributeError, KeyError):
            # Preserve benchmark execution if a non-PARTNR environment omits
            # this optional official metric.
            return False

    def get_low_level_actions(
        self, instruction: str, observations: Dict[str, Any], world_graph: Dict[int, Any]
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        # Deliberately use only `world_graph`, never `full_world_graph` or metrics.
        self.board.observe(world_graph)
        if self._official_completion_reached():
            completion_info: Dict[str, Any] = {
                "replanned": {uid: False for uid in self.planner},
                "replan_required": {uid: False for uid in self.planner},
                "responses": {},
                "is_done": {uid: True for uid in self.planner},
                "high_level_actions": {},
                "peerconsult_execution_actions": {},
                "peerconsult_action_ticket": {},
                "peerconsult_progress_signature": self.board.progress_signature(),
                "completion_adapter": "official_env_success",
            }
            self._write_event(
                {
                    "protocol": self.board.PROTOCOL,
                    "tick": self.board.tick,
                    "episode_filename": self.episode_filename,
                    "completion_adapter": "official_env_success",
                }
            )
            return {}, completion_info, True
        cards = {uid: self.board.decision_card(uid) for uid in self.planner}
        proposals: Dict[int, Dict[str, Any]] = {}
        assert isinstance(self.planner, dict)
        for uid, planner in sorted(self.planner.items()):
            if not hasattr(planner, "set_decision_card"):
                raise TypeError("PeerConsult runner requires PeerConsultZeroShotReactPlanner")
            planner.set_decision_card(cards[uid], self.board.progress_signature())
            planner.set_forced_action(self.board.required_recovery_action(uid))
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

        self._abort_stalled_actions(planner_info, low_level_actions)
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
