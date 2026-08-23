"""Pure semantic tests for the benchmark-independent PeerConsult board."""

import unittest

from habitat_llm.peer_consult.board import PeerConsultBoard


class _Node:
    def __init__(self, name):
        self.name = name


class _Graph:
    def __init__(self, objects, held_by_robot=(), held_by_human=()):
        self._objects = [_Node(name) for name in objects]
        self._held_by_robot = set(held_by_robot)
        self._held_by_human = set(held_by_human)

    def get_all_objects(self):
        return self._objects

    def get_spot_robot(self):
        return _Node("robot")

    def get_human(self):
        return _Node("human")

    def get_room_for_entity(self, entity):
        return _Node("kitchen")

    def is_object_with_agent(self, obj, agent_type):
        return (agent_type == "robot" and obj.name in self._held_by_robot) or (
            agent_type == "human" and obj.name in self._held_by_human
        )


class PeerConsultBoardTests(unittest.TestCase):
    def setUp(self):
        self.board = PeerConsultBoard()
        self.board.observe({0: _Graph(["cup"], held_by_robot=[]), 1: _Graph(["cup"])})

    def test_duplicate_new_claim_keeps_one_agent(self):
        final, reviews = self.board.review(
            {
                0: {"is_new": True, "is_done": False, "high_level_action": ("Pick", "cup", None)},
                1: {"is_new": True, "is_done": False, "high_level_action": ("Pick", "cup", None)},
            }
        )
        self.assertEqual(final[0], ("Pick", "cup", None))
        self.assertEqual(final[1], ("Wait", "", None))
        self.assertEqual(reviews[0]["reason"], "duplicate_claim")

    def test_ongoing_action_is_never_rewritten(self):
        self.board.claims["cup"] = {"agent": 0, "tick": self.board.tick}
        final, reviews = self.board.review(
            {
                0: {"is_new": False, "is_done": False, "high_level_action": ("Pick", "cup", None)},
                1: {"is_new": True, "is_done": False, "high_level_action": ("Pick", "cup", None)},
            }
        )
        self.assertEqual(final[0], ("Pick", "cup", None))
        self.assertEqual(final[1], ("Wait", "", None))
        self.assertEqual(reviews[0]["reason"], "claim_conflict")

    def test_decision_card_has_bounds_and_no_hidden_score(self):
        card = self.board.decision_card(0)
        self.assertIn("verified_goal_progress=unavailable", card)
        self.assertNotIn("task_percent_complete", card)

    def test_owner_fact_rejects_only_a_new_peer_proposal(self):
        self.board.observe(
            {
                0: _Graph(["cup"]),
                1: _Graph(["cup"], held_by_human=["cup"]),
            }
        )
        final, reviews = self.board.review(
            {
                0: {"is_new": True, "is_done": False, "high_level_action": ("Pick", "cup", None)},
                1: {"is_new": False, "is_done": False, "high_level_action": ("Pick", "cup", None)},
            }
        )
        self.assertEqual(final[0], ("Wait", "", None))
        self.assertEqual(final[1], ("Pick", "cup", None))
        self.assertEqual(reviews[0]["reason"], "ownership")

    def test_terminal_ticket_replaces_prior_ongoing_ticket_once(self):
        ongoing = {"peerconsult_action_ticket": {0: {"id": 7, "terminal": False}}}
        terminal = {"peerconsult_action_ticket": {0: {"id": 7, "terminal": True}}}
        self.board.record_execution_evidence(ongoing)
        self.board.record_execution_evidence(ongoing)
        self.board.record_execution_evidence(terminal)
        self.assertTrue(self.board.execution_evidence[(0, 7)]["terminal"])


if __name__ == "__main__":
    unittest.main()
