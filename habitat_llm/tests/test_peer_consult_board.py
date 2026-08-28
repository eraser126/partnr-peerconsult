"""Deterministic protocol tests for PARTNR PeerConsult V4."""

import unittest

from habitat_llm.peer_consult.board import PeerConsultBoard
from habitat_llm.peer_consult.partnr_adapter import canonicalize_partnr_action, execution_outcome


class _Node:
    def __init__(self, name): self.name = name


class _Graph:
    def __init__(self, objects=(), rooms=(), furniture=(), robot=(), human=()):
        self.objects, self.rooms, self.furniture = [_Node(x) for x in objects], [_Node(x) for x in rooms], [_Node(x) for x in furniture]
        self.robot, self.human = set(robot), set(human)
    def get_all_objects(self): return self.objects
    def get_all_rooms(self): return self.rooms
    def get_all_furnitures(self): return self.furniture
    def get_all_receptacles(self): return []
    def is_object_with_agent(self, obj, who): return obj.name in (self.robot if who == "robot" else self.human)


def _proposal(action, new=True):
    return {"is_new": new, "is_done": False, "high_level_action": action}


class PeerConsultV4Tests(unittest.TestCase):
    def setUp(self):
        self.board = PeerConsultBoard()
        self.graph = _Graph(["cup", "book"], ["kitchen_0", "hall_0"], ["table_0"])
        self.board.observe({0: self.graph, 1: self.graph})

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
        self.assertIn("completion_oracle=unavailable", self.board.decision_card(0))

    def test_loop_guard_rejects_exact_repeat_once(self):
        proposal = {0: _proposal(("Pick", "cup", None)), 1: _proposal(("Wait", None, None), False)}
        self.board.review(proposal)
        self.board.review(proposal)
        final, reviews, _ = self.board.review(proposal)
        self.assertEqual(final[0][0], "Wait")
        self.assertEqual(reviews[0]["reason"], "planning_loop_guard")

    def test_terminal_failure_is_retained_as_bounded_self_recovery_fact(self):
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

    def test_successful_required_navigation_clears_recovery(self):
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

    def test_successful_place_temporarily_blocks_undoing_the_relation(self):
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


if __name__ == "__main__": unittest.main()
