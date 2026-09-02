"""Deterministic protocol tests for PARTNR PeerConsult V4."""

import unittest
from unittest.mock import patch

from habitat_llm.peer_consult.board import PeerConsultBoard
from habitat_llm.peer_consult.partnr_adapter import (
    build_grounded_transport_candidates,
    build_local_action_candidates,
    canonicalize_partnr_action,
    execution_outcome,
    rewrite_held_rearrange_to_place,
)
from habitat_llm.llm.instruct.utils import actions_parser
from habitat_llm.evaluation.peer_consult_decentralized_evaluation_runner import (
    PeerConsultDecentralizedEvaluationRunner,
)
from habitat_llm.evaluation.evaluation_runner import EvaluationRunner
from habitat_llm.planner.peer_consult_zero_shot_react_planner import (
    PeerConsultZeroShotReactPlanner,
)


class _Node:
    def __init__(self, name): self.name = name


class _Graph:
    def __init__(self, objects=(), rooms=(), furniture=(), robot=(), human=(), object_locations=None):
        self.objects, self.rooms, self.furniture = [_Node(x) for x in objects], [_Node(x) for x in rooms], [_Node(x) for x in furniture]
        self.robot, self.human = set(robot), set(human)
        self.object_locations = object_locations or {}
    def get_all_objects(self): return self.objects
    def get_all_rooms(self): return self.rooms
    def get_all_furnitures(self): return self.furniture
    def get_all_receptacles(self): return []
    def is_object_with_agent(self, obj, who): return obj.name in (self.robot if who == "robot" else self.human)
    def find_furniture_for_object(self, obj):
        name = self.object_locations.get(obj.name)
        return _Node(name) if name else None
    def get_room_for_entity(self, obj):
        room = self.object_locations.get(obj.name, "").split("@")[1] if "@" in self.object_locations.get(obj.name, "") else None
        return _Node(room) if room else None


class _Measure:
    def __init__(self, value):
        self.value, self.update_calls = value, 0
    def update_metric(self, **kwargs): self.update_calls += 1
    def get_metric(self): return self.value


class _CompletionEnv:
    def __init__(self, success):
        self.task = type("Task", (), {})()
        self.task.measurements = type("Measurements", (), {})()
        self.task.measurements.measures = {
            name: _Measure(success if name == "task_state_success" else 0)
            for name in PeerConsultDecentralizedEvaluationRunner._COMPLETION_MEASURES
        }
        self.current_episode = object()


class _AbortPlanner:
    def __init__(self):
        self.reason = None

    def abort_active_action(self, reason):
        self.reason = reason


class _ParserAgent:
    def __init__(self, uid):
        self.uid = uid


def _proposal(action, new=True):
    return {"is_new": new, "is_done": False, "high_level_action": action}


class PeerConsultV4Tests(unittest.TestCase):
    def setUp(self):
        self.board = PeerConsultBoard()
        self.graph = _Graph(["cup", "book"], ["kitchen_0", "hall_0"], ["table_0"])
        self.board.observe({0: self.graph, 1: self.graph})

    def test_legacy_four_field_transport_action_gets_partnr_reference_none(self):
        actions = actions_parser(
            [_ParserAgent(0), _ParserAgent(1)],
            "Agent_0_Action: Rearrange[mug_0,on,table_1,None]\n"
            "Agent_1_Action: Place[cup_0,within,cabinet_1,None]",
        )
        self.assertEqual(actions[0], ("Rearrange", "mug_0,on,table_1,None,None", None))
        self.assertEqual(actions[1], ("Place", "cup_0,within,cabinet_1,None,None", None))

    def test_rearrange_claims_source_not_destination(self):
        final, reviews, intents = self.board.review({
            0: _proposal(("Rearrange", "cup,on,table_0,None,None", None)),
            1: _proposal(("Rearrange", "cup,within,table_0,None,None", None)),
        })
        self.assertEqual(intents[0]["resource"], "cup")
        self.assertEqual(sum(action[0] == "Rearrange" for action in final.values()), 1)
        self.assertEqual(reviews[0]["reason"], "duplicate_claim")

    def test_peer_owned_resource_is_rejected(self):
        self.board.observe({0: self.graph, 1: _Graph(["cup"], human=["cup"])})
        final, reviews, _ = self.board.review({0: _proposal(("Pick", "cup", None)), 1: _proposal(("Wait", None, None), False)})
        self.assertEqual(final[0][0], "Wait")
        self.assertEqual(reviews[0]["reason"], "physical_ownership")

    def test_terminal_success_completes_scope_and_releases_claim(self):
        _, _, intents = self.board.review({0: _proposal(("Explore", "kitchen_0", None)), 1: _proposal(("Wait", None, None), False)})
        task_id = intents[0]["task_id"]
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {"id": 1, "task_id": task_id, "terminal": True, "outcome": "terminal_success"}}})
        self.board.observe({0: self.graph, 1: self.graph})
        final, reviews, _ = self.board.review({0: _proposal(("Explore", "kitchen_0", None)), 1: _proposal(("Wait", None, None), False)})
        self.assertEqual(self.board.tasks[task_id]["status"], "completed")
        self.assertEqual(final[0][0], "Wait")
        self.assertEqual(reviews[0]["reason"], "completed_task")

    def test_late_competing_failure_cannot_reopen_completed_task(self):
        action = ("Rearrange", "cup,on,table_0,None,None", None)
        _, _, intents = self.board.review({
            0: _proposal(action), 1: _proposal(("Wait", None, None), False)
        })
        task_id = intents[0]["task_id"]
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Rearrange", "input": action[1],
            "task_id": task_id, "terminal": True,
            "outcome": "terminal_success", "response": "Successful execution!",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {1: {
            "id": 2, "agent": 1, "action": "Rearrange", "input": action[1],
            "task_id": task_id, "terminal": True,
            "outcome": "terminal_failure", "response": "Failed to pick! Not close enough to the object.",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        final, reviews, _ = self.board.review({
            0: _proposal(("Wait", None, None), False), 1: _proposal(action)
        })
        self.assertEqual(self.board.tasks[task_id]["status"], "completed")
        self.assertEqual(final[1][0], "Wait")
        self.assertEqual(reviews[0]["reason"], "completed_task")

    def test_long_transport_claim_survives_peer_boundaries(self):
        transport = ("Rearrange", "cup,on,table_0,None,None", None)
        self.board.review({0: _proposal(transport), 1: _proposal(("Wait", None, None), False)})
        for _ in range(5):
            self.board.observe({0: self.graph, 1: self.graph})
            self.board.review({0: _proposal(transport, False), 1: _proposal(("Wait", None, None))})
        self.assertIn("cup", self.board.claims)

    def test_type_prefix_is_normalized_but_private_entities_stay_private(self):
        intent = canonicalize_partnr_action(0, ("Explore", "room: kitchen_0", None), self.graph)
        self.assertEqual(intent["action"], ("Explore", "kitchen_0", None))
        self.assertNotIn("cup", self.board.decision_card(1))

    def test_structured_outcome_and_no_privileged_completion(self):
        self.assertEqual(execution_outcome({"status": "success"}, True), "terminal_success")
        self.assertIn("completion_status=not_supplied_to_planner", self.board.decision_card(0))

    def test_next_to_rejects_reference_known_to_be_on_another_furniture(self):
        graph = _Graph(
            objects=["candle", "holder"], furniture=["table_10", "table_30"],
            object_locations={"holder": "table_30"},
        )
        intent = canonicalize_partnr_action(
            0, ("Place", "candle,on,table_10,next_to,holder", None), graph
        )
        self.assertIn("must already be on destination", intent["error"])

    def test_location_sharing_is_an_explicit_optional_plugin(self):
        self.board.share_discovered_locations = True
        graph0 = _Graph(
            objects=["cup"], rooms=["kitchen_0"], furniture=["table_0"],
            object_locations={"cup": "table_0"},
        )
        self.board.observe({0: graph0, 1: _Graph()})
        self.assertIn("cup at table_0", self.board.decision_card(1))

    def test_candidate_domains_are_local_complete_and_not_ranked(self):
        candidates = build_local_action_candidates(
            0, self.graph, {"Explore", "Navigate", "Pick", "Rearrange", "Place", "Wait"}
        )
        self.assertIn("Explore[room]: any locally known room ID.", candidates)
        self.assertIn("Navigate[target]: any locally known object, furniture, or room ID.", candidates)
        self.assertIn("Rearrange[object,relation,destination,constraint,reference]", candidates)
        self.assertNotIn("book, cup", candidates)
        self.assertIn("Place[held_object,relation,destination,constraint,reference]", candidates)
        self.assertIn("held_object must be shown as held by you", candidates)
        self.assertIn("not a priority order", candidates)

    def test_grounded_transport_candidates_match_task_objects_not_unrelated_objects(self):
        graph = _Graph(
            objects=["candle_0", "candle_holder_1", "plant_container_2", "box_3"],
            furniture=["table_10", "table_30"],
        )
        candidates = build_grounded_transport_candidates(
            "Place the candle and candle holder on the white table. Move the plant to the same table.",
            graph, {"Rearrange"},
        )
        self.assertIn("Rearrange[candle_0,on,table_10,None,None]", candidates)
        self.assertIn("Rearrange[candle_holder_1,on,table_10,None,None]", candidates)
        self.assertIn("Rearrange[plant_container_2,on,table_10,None,None]", candidates)
        self.assertNotIn("Rearrange[box_3", candidates)

    def test_held_task_object_gets_place_candidate_not_rearrange(self):
        graph = _Graph(
            objects=["plant_container_2", "candle_0"],
            furniture=["table_10", "table_30"],
            robot=["plant_container_2"],
        )
        candidates = build_grounded_transport_candidates(
            "Move the plant and candle to the table.",
            graph,
            {"Rearrange", "Place"},
            agent_uid=0,
        )
        self.assertIn("Place[plant_container_2,on,table_10,None,None]", candidates)
        self.assertNotIn("Rearrange[plant_container_2", candidates)
        self.assertIn("Rearrange[candle_0,on,table_10,None,None]", candidates)

    def test_held_rearrange_is_rewritten_to_place(self):
        graph = _Graph(
            objects=["plant_container_2"], furniture=["table_10"], robot=["plant_container_2"]
        )
        action, reason = rewrite_held_rearrange_to_place(
            0, ("Rearrange", "plant_container_2,on,table_10,None,None", None), graph
        )
        self.assertEqual(action, ("Place", "plant_container_2,on,table_10,None,None", None))
        self.assertEqual(reason, "held_object_rearrange_rewritten_to_place")

    def test_peer_held_rearrange_is_not_rewritten(self):
        graph = _Graph(
            objects=["plant_container_2"], furniture=["table_10"], human=["plant_container_2"]
        )
        action, reason = rewrite_held_rearrange_to_place(
            0, ("Rearrange", "plant_container_2,on,table_10,None,None", None), graph
        )
        self.assertEqual(action[0], "Rearrange")
        self.assertIsNone(reason)

    def test_official_completion_adapter_reads_only_success_bit(self):
        runner = object.__new__(PeerConsultDecentralizedEvaluationRunner)
        curr_env = _CompletionEnv(success=1.0)
        runner.env_interface = type("Interface", (), {})()
        runner.env_interface.env = type("Outer", (), {})()
        runner.env_interface.env.env = type("Middle", (), {})()
        runner.env_interface.env.env.env = type("Inner", (), {"_env": curr_env})()
        self.assertTrue(runner._official_completion_reached())
        for measure in curr_env.task.measurements.measures.values():
            self.assertEqual(measure.update_calls, 1)

    def test_stalled_action_watchdog_aborts_and_reports_terminal_failure(self):
        runner = object.__new__(PeerConsultDecentralizedEvaluationRunner)
        planner = _AbortPlanner()
        runner.planner = {0: planner}
        runner.max_action_wall_time_s = 300
        runner._active_action_started_at = {(0, 4): 100.0}
        planner_info = {
            "peerconsult_action_ticket": {
                0: {
                    "id": 4,
                    "action": "Rearrange",
                    "input": "cup,on,table_0,None,None",
                    "terminal": False,
                    "outcome": "ongoing",
                }
            },
            "responses": {0: ""},
            "replan_required": {0: False},
            "replanned": {0: False},
            "high_level_actions": {0: ("Rearrange", "cup,on,table_0,None,None", None)},
            "peerconsult_execution_actions": {0: ("Rearrange", "cup,on,table_0,None,None", None)},
        }
        low_level_actions = {0: object()}
        with patch(
            "habitat_llm.evaluation.peer_consult_decentralized_evaluation_runner.time.monotonic",
            return_value=401.0,
        ):
            runner._abort_stalled_actions(planner_info, low_level_actions)
        ticket = planner_info["peerconsult_action_ticket"][0]
        self.assertTrue(ticket["terminal"])
        self.assertEqual(ticket["outcome"], "terminal_failure")
        self.assertIn("watchdog", ticket["response"])
        self.assertIn("watchdog", planner.reason)
        self.assertNotIn(0, low_level_actions)
        self.assertTrue(planner_info["replan_required"][0])
        self.assertNotIn(0, planner_info["peerconsult_execution_actions"])

    def test_rejected_v4_proposal_is_not_written_as_an_executed_action(self):
        runner = object.__new__(EvaluationRunner)
        runner.env_interface = type("Interface", (), {})()
        runner.env_interface.agent_action_history = {0: []}
        runner.env_interface.world_graph = {0: self.graph}
        runner.update_agent_action_history(
            {
                "replan_required": {0: True},
                "replanned": {0: True},
                "high_level_actions": {},
                "peerconsult_execution_actions": {},
                "responses": {},
                "sim_step_count": 1,
            }
        )
        self.assertEqual(runner.env_interface.agent_action_history[0], [])

    def test_loop_guard_rejects_exact_repeat_until_progress(self):
        proposal = {0: _proposal(("Pick", "cup", None)), 1: _proposal(("Wait", None, None), False)}
        self.board.review(proposal)
        self.board.review(proposal)
        final, reviews, _ = self.board.review(proposal)
        self.assertEqual(final[0][0], "Wait")
        self.assertEqual(reviews[0]["reason"], "planning_loop_guard")
        # Rejecting an identical retry must not clear the guard; otherwise
        # Qwen can alternate between an accepted retry and a rejected retry.
        final, reviews, _ = self.board.review(proposal)
        self.assertEqual(final[0][0], "Wait")
        self.assertEqual(reviews[0]["reason"], "planning_loop_guard")
        # A verified state transition permits the action again.
        self.board.progress_versions[0] += 1
        final, reviews, _ = self.board.review(proposal)
        self.assertEqual(final[0][0], "Pick")
        self.assertEqual(reviews, [])

    def test_repeated_factual_rejection_holds_without_another_replan(self):
        planner = object.__new__(PeerConsultZeroShotReactPlanner)
        planner._agents = [_ParserAgent(0)]
        planner.curr_prompt, planner.trace = "prompt", "trace"
        planner.replanning_count = 4
        planner.last_high_level_actions = {}
        planner.replan_required = True
        planner.is_done = False
        planner._peerconsult_refresh_prompt = False
        planner._peerconsult_ticket_index = 0
        planner._peerconsult_current_ticket = {}
        planner._peerconsult_progress_signature = ("before",)
        planner._peerconsult_hold_signature = None
        planner._peerconsult_last_rejection = None
        planner.get_last_agent_states = lambda: {}
        proposal = {
            "is_new": True, "is_done": False,
            "high_level_action": ("Navigate", "cup", None),
            "thought": "retry", "print": "",
        }
        review = {"verdict": "revise", "reason": "planning_loop_guard"}
        planner.execute_proposal(proposal, ("Wait", None, None), {}, review=review)
        _, info, _ = planner.execute_proposal(
            proposal, ("Wait", None, None), {}, review=review
        )
        self.assertFalse(info["replan_required"][0])
        held = planner.prepare_proposal("task", {}, {0: self.graph})
        self.assertTrue(held["hold"])
        planner.set_decision_card("new facts", ("after",))
        self.assertNotEqual(
            planner._peerconsult_hold_signature,
            planner._peerconsult_progress_signature,
        )
        self.assertTrue(planner._release_hold_after_progress())
        self.assertIsNone(planner._peerconsult_hold_signature)
        self.assertTrue(planner.replan_required)

    def test_required_recovery_is_available_as_exact_board_action(self):
        self.board.enforce_required_recovery = True
        _, _, intents = self.board.review({0: _proposal(("Pick", "cup", None)), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Pick", "input": "cup", "task_id": intents[0]["task_id"],
            "terminal": True, "outcome": "terminal_failure",
            "response": "Failed to pick! Not close enough to the object.",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        self.assertEqual(self.board.required_recovery_action(0), ("Navigate", "cup", None))

    def test_public_progress_resets_replanning_budget(self):
        planner = object.__new__(PeerConsultZeroShotReactPlanner)
        planner.replanning_count = 5
        planner._peerconsult_progress_signature = ("before",)
        planner.set_decision_card("new public fact", ("after",))
        self.assertEqual(planner.replanning_count, 0)

    def test_terminal_failure_is_retained_as_bounded_self_recovery_fact(self):
        self.board.enforce_required_recovery = True
        _, _, intents = self.board.review({0: _proposal(("Pick", "cup", None)), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Pick", "input": "cup", "task_id": intents[0]["task_id"],
            "terminal": True, "outcome": "terminal_failure",
            "response": "Unexpected failure! - Failed to pick! Not close enough to the object.",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        card = self.board.decision_card(0)
        self.assertIn("Pick[cup] failed", card)
        self.assertIn("self_required_recovery=Navigate[cup]", card)
        self.assertNotIn("Pick[cup] failed", self.board.decision_card(1))
        self.assertNotIn("cup: needs_recovery", self.board.decision_card(1))

    def test_distance_failure_adds_private_advisory_recovery_without_forcing_it(self):
        _, _, intents = self.board.review({0: _proposal(("Pick", "cup", None)), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Pick", "input": "cup", "task_id": intents[0]["task_id"],
            "terminal": True, "outcome": "terminal_failure",
            "response": "Failed to pick: Not close enough to the object.",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        self.assertIn("self_recovery_advice=Navigate[cup]", self.board.decision_card(0))
        self.assertIn("self_required_recovery=none", self.board.decision_card(0))
        self.assertNotIn("Navigate[cup]", self.board.decision_card(1))

    def test_successful_required_navigation_clears_recovery(self):
        self.board.enforce_required_recovery = True
        self.board.required_recoveries[0] = "Navigate[cup]"
        _, _, intents = self.board.review({0: _proposal(("Navigate", "cup", None)), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Navigate", "input": "cup", "task_id": intents[0]["task_id"],
            "terminal": True, "outcome": "terminal_success", "response": "Successful execution!",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        self.assertIn("self_required_recovery=none", self.board.decision_card(0))

    def test_same_room_explore_is_atomically_reserved(self):
        final, reviews, _ = self.board.review({
            0: _proposal(("Explore", "kitchen_0", None)),
            1: _proposal(("Explore", "kitchen_0", None)),
        })
        self.assertEqual(sum(action[0] == "Explore" for action in final.values()), 1)
        self.assertEqual(sum(action[0] == "Wait" for action in final.values()), 1)
        self.assertEqual(reviews[0]["reason"], "duplicate_room_reservation")

    def test_place_distance_failure_requires_navigation_before_next_action(self):
        self.board.enforce_required_recovery = True
        place = ("Place", "cup,on,table_0,None,None", None)
        _, _, intents = self.board.review({0: _proposal(place), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Place", "input": "cup,on,table_0,None,None",
            "task_id": intents[0]["task_id"], "terminal": True, "outcome": "terminal_failure",
            "response": "Unexpected failure! - Failed to place! Not close enough to table_0 or occluded.",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        final, reviews, _ = self.board.review({0: _proposal(("Pick", "cup", None)), 1: _proposal(("Wait", None, None), False)})
        self.assertEqual(final[0][0], "Wait")
        self.assertEqual(reviews[0]["reason"], "required_recovery")
        final, _, _ = self.board.review({0: _proposal(("Navigate", "table_0", None)), 1: _proposal(("Wait", None, None), False)})
        self.assertEqual(final[0][0], "Navigate")

    def test_rearrange_pick_distance_failure_recovers_to_source_object(self):
        self.board.enforce_required_recovery = True
        action = ("Rearrange", "cup,on,table_0,None,None", None)
        _, _, intents = self.board.review({0: _proposal(action), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Rearrange", "input": action[1],
            "task_id": intents[0]["task_id"], "terminal": True,
            "outcome": "terminal_failure",
            "response": "Unexpected failure! - Failed to pick! Not close enough to the object.",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        self.assertIn("self_required_recovery=Navigate[cup]", self.board.decision_card(0))

    def test_successful_place_temporarily_blocks_undoing_the_relation(self):
        self.board.enforce_placement_stability = True
        place = ("Place", "cup,on,table_0,None,None", None)
        _, _, intents = self.board.review({0: _proposal(place), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Place", "input": "cup,on,table_0,None,None",
            "task_id": intents[0]["task_id"], "terminal": True, "outcome": "terminal_success",
            "response": "Successful execution!",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        final, reviews, _ = self.board.review({0: _proposal(("Wait", None, None), False), 1: _proposal(("Pick", "cup", None))})
        self.assertEqual(final[1][0], "Wait")
        self.assertEqual(reviews[0]["reason"], "settled_relation")

    def test_exploration_report_is_durable_shared_memory(self):
        self.board.publish_exploration_reports = True
        _, _, intents = self.board.review({0: _proposal(("Explore", "kitchen_0", None)), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Explore", "input": "kitchen_0", "task_id": intents[0]["task_id"],
            "terminal": True, "outcome": "terminal_success",
            "response": "Successful execution!\nObjects:\ncup: table_0 in kitchen_0",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        card = self.board.decision_card(1)
        self.assertIn("kitchen_0 by agent0: cup: table_0 in kitchen_0", card)
        self.assertIn("self_available_unexplored_rooms=['hall_0']", card)

    def test_peer_cannot_repeat_completed_room_task(self):
        self.board.publish_exploration_reports = True
        self.board.enforce_room_exploration_dedup = True
        _, _, intents = self.board.review({0: _proposal(("Explore", "kitchen_0", None)), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Explore", "input": "kitchen_0", "task_id": intents[0]["task_id"],
            "terminal": True, "outcome": "terminal_success", "response": "Successful execution!\nObjects:\ncup: table_0",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        final, reviews, _ = self.board.review({0: _proposal(("Wait", None, None), False), 1: _proposal(("Explore", "kitchen_0", None))})
        self.assertEqual(final[1][0], "Wait")
        self.assertEqual(reviews[0]["reason"], "completed_task")
        self.assertIn("Advance to a different unsatisfied", self.board.decision_card(1))

    def test_successful_transport_is_retained_in_public_object_ledger(self):
        action = ("Rearrange", "cup,on,table_0,None,None", None)
        _, _, intents = self.board.review({0: _proposal(action), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Rearrange", "input": "cup,on,table_0,None,None",
            "task_id": intents[0]["task_id"], "terminal": True, "outcome": "terminal_success",
            "response": "Successful execution!",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        self.assertIn("cup: placed by agent0 (on table_0)", self.board.decision_card(1))

    def test_v4_baseline_keeps_private_explore_observation_private(self):
        _, _, intents = self.board.review({0: _proposal(("Explore", "kitchen_0", None)), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Explore", "input": "kitchen_0", "task_id": intents[0]["task_id"],
            "terminal": True, "outcome": "terminal_success",
            "response": "Successful execution!\\nObjects:\\ncup: table_0 in kitchen_0",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        card = self.board.decision_card(1)
        self.assertNotIn("cup: table_0 in kitchen_0", card)
        self.assertIn("self_available_unexplored_rooms=['hall_0']", card)

    def test_progress_signature_changes_only_for_positive_public_facts(self):
        before = self.board.progress_signature()
        _, _, intents = self.board.review({0: _proposal(("Pick", "cup", None)), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 1, "action": "Pick", "input": "cup", "task_id": intents[0]["task_id"],
            "terminal": True, "outcome": "terminal_failure", "response": "Failed to pick.",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        self.assertEqual(before, self.board.progress_signature())

        _, _, intents = self.board.review({0: _proposal(("Explore", "kitchen_0", None)), 1: _proposal(("Wait", None, None), False)})
        self.board.record_execution_evidence({"peerconsult_action_ticket": {0: {
            "id": 2, "action": "Explore", "input": "kitchen_0", "task_id": intents[0]["task_id"],
            "terminal": True, "outcome": "terminal_success", "response": "Successful execution!",
        }}})
        self.board.observe({0: self.graph, 1: self.graph})
        self.assertNotEqual(before, self.board.progress_signature())

    def test_task_ids_are_agent_independent_and_navigate_does_not_reserve_room(self):
        delivery0 = canonicalize_partnr_action(0, ("Rearrange", "cup,on,table_0,None,None", None), self.graph)
        delivery1 = canonicalize_partnr_action(1, ("Rearrange", "cup,on,table_0,None,None", None), self.graph)
        navigation = canonicalize_partnr_action(1, ("Navigate", "kitchen_0", None), self.graph)
        self.assertEqual(delivery0["task_id"], delivery1["task_id"])
        self.assertEqual(navigation["stage"], "navigate")
        self.assertIsNone(navigation["room_scope"])
        self.assertNotEqual(navigation["task_id"], "room:kitchen_0")


if __name__ == "__main__": unittest.main()
