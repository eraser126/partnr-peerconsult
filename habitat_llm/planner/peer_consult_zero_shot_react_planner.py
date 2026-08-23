"""Proposal/execution split for the PARTNR PeerConsult variant.

The original ``ZeroShotReactPlanner`` remains unchanged.  This subclass uses
the same model, grammar, parser and skills, but exposes a proposal boundary so
the coordinator can review *new* high-level actions before any motor skill is
started.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Tuple, Union

from habitat_llm.llm.instruct.utils import get_objects_descr
from habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner


class PeerConsultZeroShotReactPlanner(ZeroShotReactPlanner):
    def reset(self) -> None:
        super().reset()
        self.peerconsult_card = "[PeerConsult Decision Card]\nNo shared facts yet."
        self._peerconsult_ticket_index = 0
        self._peerconsult_current_ticket: Dict[str, Any] = {}

    def set_decision_card(self, card: str) -> None:
        self.peerconsult_card = card

    def prepare_prompt(self, input_instruction, world_graph, **kwargs):
        _, params = super().prepare_prompt(
            input_instruction, world_graph, should_format=False, **kwargs
        )
        params["peerconsult_card"] = self.peerconsult_card
        return self.prompt.format(**params), params

    def _append_card_for_next_turn(self) -> None:
        """Insert a new bounded card at a legal user-turn boundary."""
        assistant_thought = self.planner_config.llm.assistant_tag + "Thought:"
        if self.curr_prompt.endswith(assistant_thought):
            self.curr_prompt = self.curr_prompt[: -len(assistant_thought)]
        card_turn = (
            self.planner_config.llm.user_tag
            + self.peerconsult_card
            + "\n"
            + self.planner_config.llm.eot_tag
            + self.planner_config.llm.assistant_tag
            + "Thought:"
        )
        self.curr_prompt += card_turn
        self.trace += "\n" + self.peerconsult_card + "\nThought:"

    def prepare_proposal(self, instruction, observations, world_graph, verbose=False):
        """Generate and parse a new action, but do not start its skill yet."""
        if self.is_done:
            return {
                "is_new": False,
                "is_done": True,
                "high_level_action": ("Done", None, None),
                "thought": None,
                "print": "",
            }

        if self.curr_prompt == "":
            self.curr_prompt, self.params = self.prepare_prompt(
                instruction, world_graph[self._agents[0].uid], observations=observations
            )
            self.curr_obj_states = get_objects_descr(
                world_graph[self._agents[0].uid],
                self._agents[0].uid,
                include_room_name=True,
                add_state_info=self.planner_config.objects_response_include_states,
                centralized=self.planner_config.centralized,
            )
        if self.trace == "":
            self.trace += "Task: {}\nThought: ".format(instruction)

        agent_uid = self._agents[0].uid
        if not self.replan_required:
            return {
                "is_new": False,
                "is_done": False,
                "high_level_action": self.last_high_level_actions[agent_uid],
                "thought": None,
                "print": "",
            }

        if verbose:
            start_time = time.time()
        response_info = self.replan(instruction, observations, world_graph)
        llm_response = response_info["llm_response"]
        thought = self.parse_thought(llm_response)
        if verbose:
            print("Time taken for LLM response generation: {}".format(time.time() - start_time))

        print_str = "{}\n{}\n".format(llm_response, self.stopword)
        prompt_addition = "{}\n{}{}".format(
            llm_response, self.stopword, self.planner_config.llm.eot_tag
        )
        self.curr_prompt += prompt_addition
        self.trace += prompt_addition
        done_by_model = self.check_if_agent_done(llm_response)
        done_by_limit = (
            self.replanning_count == self.planner_config.replanning_threshold
        )
        self.is_done = done_by_model or done_by_limit
        self.replanning_count += 1
        if self.is_done:
            return {
                "is_new": True,
                "is_done": True,
                "high_level_action": ("Done", None, None),
                "thought": thought,
                "print": print_str,
                "done_reason": "replanning_limit" if done_by_limit else "model_done",
            }

        high_level_actions = self.actions_parser(self.agents, llm_response, self.params)
        action = high_level_actions.get(
            agent_uid,
            (None, None, "No action could be parsed from the LLM response."),
        )
        return {
            "is_new": True,
            "is_done": False,
            "high_level_action": action,
            "thought": thought,
            "print": print_str,
            "done_reason": None,
        }

    def execute_proposal(self, proposal, final_action, observations):
        """Execute a reviewed proposal through the unchanged PARTNR skill API."""
        agent_uid = self._agents[0].uid
        is_new = bool(proposal["is_new"])
        is_done = bool(proposal["is_done"])
        print_str = proposal["print"]
        if is_done:
            # The board may reject Done[] only from grounded local evidence
            # (held object or immediately preceding skill failure). Continue
            # the same ReAct transcript rather than ending the episode.
            done_rejection = final_action[2]
            if done_rejection:
                self.is_done = False
                self.replan_required = True
                responses = {agent_uid: str(done_rejection)}
                print_str += self._add_responses_to_prompt(responses)
                self._append_card_for_next_turn()
                self._peerconsult_ticket_index += 1
                ticket = {
                    "id": self._peerconsult_ticket_index,
                    "action": "Done",
                    "input": None,
                    "valid": False,
                    "terminal": True,
                    "status": "rejected",
                    "response": str(done_rejection),
                    "started_after_replan": is_new,
                    "replan_required_before_execution": True,
                }
                planner_info: Dict[str, Any] = {
                    "replanned": {agent_uid: is_new},
                    "replan_required": {agent_uid: True},
                    "responses": responses,
                    "thought": {agent_uid: proposal["thought"]},
                    "is_done": {agent_uid: False},
                    "print": print_str,
                    "high_level_actions": {agent_uid: final_action},
                    "prompts": {agent_uid: self.curr_prompt},
                    "traces": {agent_uid: self.trace},
                    "replanning_count": {agent_uid: self.replanning_count},
                    "agent_states": self.get_last_agent_states(),
                    "peerconsult_action_ticket": {agent_uid: ticket},
                }
                return {}, planner_info, False
            return {}, self._done_info(proposal), True

        low_level_actions, responses = self.process_high_level_actions(
            {agent_uid: final_action}, observations
        )
        if is_new:
            self.last_high_level_actions = {agent_uid: final_action}
            self._peerconsult_ticket_index += 1
            self._peerconsult_current_ticket = {
                "id": self._peerconsult_ticket_index,
                "action": final_action[0],
                "input": final_action[1],
                "valid": final_action[2] is None,
            }

        previous_replan_required = self.replan_required
        self.replan_required = any(responses.values())
        print_str += self._add_responses_to_prompt(responses)
        if self.replan_required:
            self._append_card_for_next_turn()

        ticket = dict(self._peerconsult_current_ticket)
        ticket.update(
            {
                "terminal": bool(responses.get(agent_uid)),
                "status": "terminal" if responses.get(agent_uid) else "ongoing",
                "response": responses.get(agent_uid, ""),
                "started_after_replan": is_new,
                "replan_required_before_execution": previous_replan_required,
            }
        )
        planner_info: Dict[str, Any] = {
            "replanned": {agent_uid: is_new},
            "replan_required": {agent_uid: previous_replan_required},
            "responses": responses,
            "thought": {agent_uid: proposal["thought"]},
            "is_done": {agent_uid: self.is_done},
            "print": print_str,
            "high_level_actions": {agent_uid: final_action},
            "prompts": {agent_uid: self.curr_prompt},
            "traces": {agent_uid: self.trace},
            "replanning_count": {agent_uid: self.replanning_count},
            "agent_states": self.get_last_agent_states(),
            "peerconsult_action_ticket": {agent_uid: ticket},
        }
        return low_level_actions, planner_info, self.is_done

    def _done_info(self, proposal):
        agent_uid = self._agents[0].uid
        return {
            "print": proposal["print"],
            "prompts": {agent_uid: self.curr_prompt},
            "traces": {agent_uid: self.trace},
            "replanning_count": {agent_uid: self.replanning_count},
            "replan_required": {agent_uid: self.replan_required},
            "replanned": {agent_uid: bool(proposal["is_new"])},
            "is_done": {agent_uid: True},
            "thought": {agent_uid: proposal["thought"]},
            "high_level_actions": {agent_uid: ("Done", None, None)},
            "peerconsult_action_ticket": {},
        }
