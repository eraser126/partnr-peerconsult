"""V2-prompt PARTNR planner with a PeerConsult V4 proposal boundary."""

from __future__ import annotations

import time
from typing import Any, Dict

from habitat_llm.llm.instruct.utils import get_objects_descr
from habitat_llm.peer_consult.partnr_adapter import (
    build_local_action_candidates,
    canonicalize_partnr_action,
    execution_outcome,
)
from habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner


class PeerConsultZeroShotReactPlanner(ZeroShotReactPlanner):
    """Keep V2 world-description/tool prompts; add only factual V4 governance."""

    def reset(self) -> None:
        super().reset()
        self.peerconsult_card = "[PeerConsult V4 Decision Card]\nNo public facts yet."
        self._peerconsult_ticket_index = 0
        self._peerconsult_current_ticket: Dict[str, Any] = {}
        self._peerconsult_refresh_prompt = False

    def set_decision_card(self, card: str) -> None:
        self.peerconsult_card = card

    def prepare_prompt(self, input_instruction, world_graph, **kwargs):
        _, params = super().prepare_prompt(input_instruction, world_graph, should_format=False, **kwargs)
        params["peerconsult_card"] = self.peerconsult_card
        params["local_action_candidates"] = build_local_action_candidates(
            self._agents[0].uid, world_graph, self._agents[0].tools.keys()
        )
        return self.prompt.format(**params), params

    def _fresh_prompt(self, instruction, observations, graph) -> None:
        """Rebuild from current local observation instead of accumulating LLM history."""
        self.curr_prompt, self.params = self.prepare_prompt(instruction, graph, observations=observations)
        self.curr_obj_states = get_objects_descr(
            graph,
            self._agents[0].uid,
            include_room_name=True,
            add_state_info=self.planner_config.objects_response_include_states,
            centralized=self.planner_config.centralized,
        )
        self._peerconsult_refresh_prompt = False

    def _validated_action(self, llm_response: str, graph: Any):
        actions = self.actions_parser(self.agents, llm_response, self.params)
        uid = self._agents[0].uid
        action = actions.get(uid, (None, None, "No action could be parsed from the LLM response."))
        intent = canonicalize_partnr_action(uid, action, graph)
        if intent.get("error"):
            return (None, None, str(intent["error"])), None
        return tuple(intent["action"]), intent

    def prepare_proposal(self, instruction, observations, world_graph, verbose=False):
        uid = self._agents[0].uid
        if self.is_done:
            return {"is_new": False, "is_done": True, "high_level_action": ("Done", None, None), "thought": None, "print": ""}
        if self.curr_prompt == "" or self._peerconsult_refresh_prompt:
            self._fresh_prompt(instruction, observations, world_graph[uid])
        if self.trace == "":
            self.trace = "Task: {}\nThought: ".format(instruction)
        if not self.replan_required:
            return {"is_new": False, "is_done": False, "high_level_action": self.last_high_level_actions[uid], "thought": None, "print": "", "intent": None}
        start = time.time() if verbose else None
        response_info = self.replan(instruction, observations, world_graph)
        llm_response = response_info["llm_response"]
        if verbose:
            print("Time taken for LLM response generation: {}".format(time.time() - start))
        thought = self.parse_thought(llm_response)
        print_str = "{}\n{}\n".format(llm_response, self.stopword)
        self.curr_prompt += "{}\n{}{}".format(llm_response, self.stopword, self.planner_config.llm.eot_tag)
        self.trace += "{}\n{}{}".format(llm_response, self.stopword, self.planner_config.llm.eot_tag)
        self.replanning_count += 1
        if self.replanning_count - 1 == self.planner_config.replanning_threshold:
            action, intent = ("Done", None, None), canonicalize_partnr_action(uid, ("Done", None, None), world_graph[uid])
        else:
            action, intent = self._validated_action(llm_response, world_graph[uid])
        return {"is_new": True, "is_done": False, "high_level_action": action, "thought": thought, "print": print_str, "intent": intent}

    def _ticket(self, action, proposal, response, terminal, task_id=None):
        if proposal.get("is_new"):
            self._peerconsult_ticket_index += 1
            self._peerconsult_current_ticket = {
                "id": self._peerconsult_ticket_index, "action": action[0], "input": action[1],
                "task_id": task_id, "valid": action[2] is None,
            }
        ticket = dict(self._peerconsult_current_ticket)
        ticket.update({"terminal": terminal, "status": "terminal" if terminal else "ongoing", "response": response, "outcome": execution_outcome(response, terminal)})
        return ticket

    def execute_proposal(self, proposal, final_action, observations, review=None, intent=None):
        """Execute accepted actions only; invalid/rejected proposals stay private."""
        uid, review, intent = self._agents[0].uid, (review or {}), (intent or {})
        if proposal.get("is_done"):
            return {}, self._done_info(proposal), True
        rejected = review.get("verdict") == "revise"
        parser_rejected = bool(final_action[2])
        if rejected or parser_rejected:
            self.last_high_level_actions = {}
            self.replan_required = True
            self._peerconsult_refresh_prompt = True
            reason = review.get("reason") if rejected else final_action[2]
            response = "validator: revise ({})".format(reason)
            ticket = self._ticket(final_action, proposal, response, True, intent.get("task_id"))
            return {}, {
                # A validator rejection follows a fresh LLM proposal.  Mark it
                # as such so the runner's no-progress guard counts reject loops.
                "replanned": {uid: bool(proposal.get("is_new"))}, "replan_required": {uid: True}, "responses": {},
                "thought": {uid: proposal.get("thought")}, "is_done": {uid: False},
                "print": proposal.get("print", ""), "high_level_actions": {}, "peerconsult_execution_actions": {},
                "prompts": {uid: self.curr_prompt}, "traces": {uid: self.trace},
                "replanning_count": {uid: self.replanning_count}, "agent_states": self.get_last_agent_states(),
                "peerconsult_action_ticket": {uid: ticket},
            }, False
        self.last_high_level_actions = {uid: final_action}
        low_level_actions, responses = self.process_high_level_actions({uid: final_action}, observations)
        response = responses.get(uid, "")
        self.replan_required = any(responses.values())
        if self.replan_required:
            self._peerconsult_refresh_prompt = True
        ticket = self._ticket(final_action, proposal, response, bool(response), intent.get("task_id"))
        return low_level_actions, {
            "replanned": {uid: bool(proposal.get("is_new"))}, "replan_required": {uid: self.replan_required},
            "responses": responses, "thought": {uid: proposal.get("thought")}, "is_done": {uid: self.is_done},
            "print": proposal.get("print", "") + self._add_responses_to_prompt(responses),
            "high_level_actions": {uid: final_action}, "peerconsult_execution_actions": {uid: final_action},
            "prompts": {uid: self.curr_prompt}, "traces": {uid: self.trace},
            "replanning_count": {uid: self.replanning_count}, "agent_states": self.get_last_agent_states(),
            "peerconsult_action_ticket": {uid: ticket},
        }, self.is_done

    def _done_info(self, proposal):
        uid = self._agents[0].uid
        return {"print": proposal.get("print", ""), "prompts": {uid: self.curr_prompt}, "traces": {uid: self.trace}, "replanning_count": {uid: self.replanning_count}, "replan_required": {uid: self.replan_required}, "replanned": {uid: bool(proposal.get("is_new"))}, "is_done": {uid: True}, "thought": {uid: proposal.get("thought")}, "high_level_actions": {uid: ("Done", None, None)}, "peerconsult_action_ticket": {}}
