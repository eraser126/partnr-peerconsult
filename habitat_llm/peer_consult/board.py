"""Bounded, evidence-oriented coordination state for PARTNR PeerConsult.

This module must not read ``EnvironmentInterface.full_world_graph`` or task
evaluation propositions.  Those are hidden privileged sources in the
partial-observation benchmark.  All facts below originate in an individual
agent's existing world graph or from that agent's skill feedback.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


Action = Tuple[Optional[str], Optional[str], Optional[str]]


class PeerConsultBoard:
    """Shared symbolic board with small, agent-specific decision cards.

    It is intentionally conservative: ownership is only accepted from the
    owner agent's own graph; task completion stays unavailable during an
    episode because PARTNR exposes it through hidden evaluator propositions.
    """

    def __init__(
        self,
        claim_ttl_decisions: int = 3,
        max_targets: int = 5,
        max_rooms: int = 4,
        max_reviews: int = 2,
        max_failed_action_retries: int = 1,
        done_veto_on_recent_failure: bool = True,
    ) -> None:
        self.claim_ttl_decisions = claim_ttl_decisions
        self.max_targets = max_targets
        self.max_rooms = max_rooms
        self.max_reviews = max_reviews
        self.max_failed_action_retries = max_failed_action_retries
        self.done_veto_on_recent_failure = done_veto_on_recent_failure
        self.reset()

    def reset(self) -> None:
        self.tick = 0
        self.known_entities: Dict[str, Dict[str, Any]] = {}
        self.known_furniture: Dict[str, Dict[str, Any]] = {}
        self.agent_rooms: Dict[int, str] = {}
        self.physical_owners: Dict[str, int] = {}
        self.claims: Dict[str, Dict[str, Any]] = {}
        self.current_intents: Dict[int, Dict[str, Any]] = {}
        self.reviews: List[Dict[str, Any]] = []
        self.execution_evidence: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.failure_counts: Dict[Tuple[str, str], int] = {}
        self.last_terminal_feedback: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _safe_room(graph: Any, entity: Any) -> str:
        try:
            room = graph.get_room_for_entity(entity)
            return str(room.name)
        except Exception:
            return "unknown"

    @staticmethod
    def _agent_node(graph: Any, agent_uid: int) -> Any:
        try:
            return graph.get_spot_robot() if agent_uid == 0 else graph.get_human()
        except Exception:
            return None

    @staticmethod
    def _owner_from_own_graph(graph: Any, obj: Any, agent_uid: int) -> Optional[int]:
        """Return only a self-owned fact, never a hidden global ownership fact."""
        try:
            if agent_uid == 0 and graph.is_object_with_agent(obj, "robot"):
                return 0
            if agent_uid == 1 and graph.is_object_with_agent(obj, "human"):
                return 1
        except Exception:
            pass
        return None

    def _expire_claims(self) -> None:
        stale = [
            target
            for target, claim in self.claims.items()
            if self.tick - int(claim["tick"]) > self.claim_ttl_decisions
            or target in self.physical_owners
        ]
        for target in stale:
            self.claims.pop(target, None)

    def observe(self, world_graphs: Mapping[int, Any]) -> None:
        """Reconcile only facts already available to the two baseline agents."""
        self.tick += 1
        self.agent_rooms = {}
        self.physical_owners = {}

        for agent_uid, graph in world_graphs.items():
            agent_node = self._agent_node(graph, agent_uid)
            if agent_node is not None:
                self.agent_rooms[agent_uid] = self._safe_room(graph, agent_node)
            else:
                self.agent_rooms[agent_uid] = "unknown"

            try:
                objects: Iterable[Any] = graph.get_all_objects()
            except Exception:
                objects = []
            for obj in objects:
                object_id = str(obj.name)
                record = self.known_entities.setdefault(
                    object_id,
                    {
                        "id": object_id,
                        "kind": "object",
                        "first_seen_tick": self.tick,
                        "sources": [],
                    },
                )
                record["last_seen_tick"] = self.tick
                if agent_uid not in record["sources"]:
                    record["sources"].append(agent_uid)
                room = self._safe_room(graph, obj)
                if room != "unknown":
                    record["room"] = room
                owner = self._owner_from_own_graph(graph, obj, agent_uid)
                if owner is not None:
                    self.physical_owners[object_id] = owner

            # Furniture facts come from the same partial graph as the original
            # agent grammar, never from a full world graph.
            try:
                furnitures: Iterable[Any] = graph.get_all_furnitures()
            except Exception:
                furnitures = []
            for furniture in furnitures:
                furniture_id = str(furniture.name)
                record = self.known_furniture.setdefault(
                    furniture_id,
                    {
                        "id": furniture_id,
                        "kind": "furniture",
                        "first_seen_tick": self.tick,
                        "sources": [],
                    },
                )
                record["last_seen_tick"] = self.tick
                if agent_uid not in record["sources"]:
                    record["sources"].append(agent_uid)
                room = self._safe_room(graph, furniture)
                if room != "unknown":
                    record["room"] = room

        self._expire_claims()

    def _held_by(self, agent_uid: int) -> List[str]:
        return sorted(
            object_id
            for object_id, owner in self.physical_owners.items()
            if owner == agent_uid
        )

    def decision_card(self, agent_uid: int) -> str:
        """Render a bounded prompt fragment; never serialize all board history."""
        peer_uid = 1 if agent_uid == 0 else 0
        held = self._held_by(agent_uid)
        peer_held = self._held_by(peer_uid)
        candidates = [
            entity
            for entity in self.known_entities.values()
            if entity["id"] not in self.physical_owners
            and self.claims.get(entity["id"], {}).get("agent") != peer_uid
        ]
        candidates.sort(
            key=lambda entity: (
                entity.get("room") != self.agent_rooms.get(agent_uid, "unknown"),
                -int(entity.get("last_seen_tick", 0)),
                entity["id"],
            )
        )
        candidate_lines = [
            "- {} | room={} | last_seen={}".format(
                entity["id"],
                entity.get("room", "unknown"),
                entity.get("last_seen_tick", "unknown"),
            )
            for entity in candidates[: self.max_targets]
        ]
        room_names = sorted(
            {
                str(entity.get("room"))
                for entity in self.known_entities.values()
                if entity.get("room") and entity.get("room") != "unknown"
            }
        )[: self.max_rooms]
        peer_claims = [
            target
            for target, claim in sorted(self.claims.items())
            if claim["agent"] == peer_uid
        ][: self.max_targets]
        relevant_reviews = [
            review
            for review in reversed(self.reviews)
            if review["agent"] == agent_uid
        ][: self.max_reviews]

        return "\n".join(
            [
                "[PeerConsult Decision Card]",
                "coordination_tick={}".format(self.tick),
                "verified_goal_progress=unavailable (hidden evaluator is not used)",
                "self: room={} held={}".format(
                    self.agent_rooms.get(agent_uid, "unknown"), held or ["none"]
                ),
                "peer: room={} held={} intent={}".format(
                    self.agent_rooms.get(peer_uid, "unknown"),
                    peer_held or ["none"],
                    self.current_intents.get(peer_uid, {}).get("stage", "idle"),
                ),
                "peer_claims={}".format(peer_claims or ["none"]),
                "candidate_objects:",
                *(candidate_lines or ["- none known"]),
                "known_rooms={}".format(room_names or ["none"]),
                "recent_reviews={}".format(
                    [review["reason"] for review in relevant_reviews] or ["none"]
                ),
                "spatial_action_rule: for a fixed chair/sofa/bed/table reference, use "
                "Rearrange[object, on, floor, next_to, reference]. Do not use "
                "within floor. For a movable reference, it must already be on the "
                "same target furniture.",
                "termination_rule: do not use Done[] while holding an object or immediately "
                "after a failed skill. Inspect the latest local observation and recover first.",
                "Use this card only as grounded coordination context. Do not state that a task is complete without environment evidence.",
            ]
        )

    @staticmethod
    def _action_args(action: Action) -> List[str]:
        """Split a public high-level action argument string conservatively."""
        action_input = action[1]
        if action_input is None:
            return []
        return [part.strip() for part in str(action_input).split(",")]

    @classmethod
    def _action_signature(cls, action: Action) -> Tuple[str, str]:
        return ((action[0] or "").lower(), ",".join(cls._action_args(action)).lower())

    @staticmethod
    def _is_none(value: str) -> bool:
        return value.strip().lower() in {"", "none"}

    def _spatial_action_error(self, action: Action) -> Optional[str]:
        """Reject only malformed or known-infeasible placement requests.

        This is a method-side API guard. It uses no task propositions and does
        not infer episode success; it only encodes the public Place/Rearrange
        contract before an expensive motor skill starts.
        """
        name = (action[0] or "").lower()
        if name not in {"place", "rearrange"} or action[2] is not None:
            return None
        args = self._action_args(action)
        if len(args) != 5:
            return None  # Let the original skill report its own syntax error.
        _, relation, receptacle, constraint, reference = args
        relation = relation.lower()
        receptacle = receptacle.lower()
        constraint = constraint.lower()
        has_constraint = not self._is_none(constraint)
        has_reference = not self._is_none(reference)

        if relation not in {"on", "within"}:
            return (
                "PeerConsult action check: Place/Rearrange relation must be 'on' or "
                "'within'. Re-plan using the documented five-argument API."
            )
        if receptacle == "floor" and relation != "on":
            return (
                "PeerConsult action check: floor supports only relation 'on'. Use "
                "Rearrange[object, on, floor, next_to, reference] when appropriate."
            )
        if has_constraint != has_reference:
            return (
                "PeerConsult action check: spatial_constraint and reference_object must "
                "either both be supplied or both be None."
            )
        if has_constraint and constraint != "next_to":
            return "PeerConsult action check: the only supported spatial constraint is 'next_to'."

        # Fixed-furniture next_to is implemented by sampling a floor pose near
        # that furniture.  Putting an object on a chair and next_to the same
        # chair is neither the intended relation nor reliably feasible.
        if has_constraint and reference in self.known_furniture and receptacle != "floor":
            return (
                "PeerConsult action check: '{}' is a fixed furniture reference. "
                "Use [object, on, floor, next_to, {}] instead of placing on/within "
                "another receptacle.".format(reference, reference)
            )
        return None

    def _done_rejection(self, agent_uid: int) -> Optional[str]:
        """Return a local-evidence reason to continue, never a hidden goal check."""
        held = self._held_by(agent_uid)
        if held:
            return (
                "PeerConsult completion check: you still locally observe that you hold "
                "{}. Place it safely and inspect the result before Done[].".format(held)
            )
        if self.done_veto_on_recent_failure:
            feedback = self.last_terminal_feedback.get(agent_uid)
            if feedback and feedback.get("failed"):
                return (
                    "PeerConsult completion check: the most recent skill failed: {}. "
                    "Recover from that local feedback and inspect the environment before "
                    "Done[].".format(feedback.get("response", "unknown failure"))
                )
        return None

    def _intent_from_action(self, action: Action) -> Dict[str, Any]:
        action_name, action_input, _ = action
        name = (action_name or "").lower()
        if name in {"pick", "rearrange"}:
            stage = "acquire"
        elif name in {"place", "clean", "fill", "pour", "open", "close", "poweron", "poweroff"}:
            stage = "manipulate"
        elif name in {"navigate", "explore"}:
            stage = "explore"
        elif name == "wait":
            stage = "wait"
        elif name == "done":
            stage = "done"
        else:
            stage = "idle"
        # For Pick/Place/Rearrange the first argument is the moved object. The
        # old whole-string key did not prevent duplicate-object claims.
        args = self._action_args(action)
        if name in {"pick", "place", "rearrange"}:
            target = args[0] if args else None
        elif name in {"navigate", "explore"}:
            target = args[0] if args else None
        else:
            target = str(action_input) if action_input else None
        if target not in self.known_entities:
            target = None
        return {"stage": stage, "target": target, "action": action_name or ""}

    @staticmethod
    def _wait_action() -> Action:
        return ("Wait", "", None)

    def review(
        self, proposals: Mapping[int, Mapping[str, Any]]
    ) -> Tuple[Dict[int, Action], List[Dict[str, Any]]]:
        """Apply evidence-triggered, one-shot, non-interrupting governance."""
        final_actions: Dict[int, Action] = {
            uid: tuple(proposal["high_level_action"])  # type: ignore[arg-type]
            for uid, proposal in proposals.items()
        }
        reviews: List[Dict[str, Any]] = []

        # Ongoing skills are preserved exactly: a review may only alter a new proposal.
        mutable = {
            uid
            for uid, proposal in proposals.items()
            if bool(proposal.get("is_new")) and not bool(proposal.get("is_done"))
        }
        intents = {
            uid: self._intent_from_action(action)
            for uid, action in final_actions.items()
        }

        # Reject model-only completion when the agent's own partial evidence
        # proves it is holding an object or its latest skill failed. No
        # evaluator state is queried.
        for uid, proposal in proposals.items():
            if not bool(proposal.get("is_new")) or not bool(proposal.get("is_done")):
                continue
            if proposal.get("done_reason") == "replanning_limit":
                continue
            reason = self._done_rejection(uid)
            if reason is not None:
                final_actions[uid] = (None, None, reason)
                intents[uid] = self._intent_from_action(final_actions[uid])
                reviews.append(
                    {
                        "tick": self.tick,
                        "agent": uid,
                        "reason": "premature_done",
                        "replacement": "replan",
                    }
                )

        # A simultaneous duplicate claim has a deterministic winner.
        targets: Dict[str, List[int]] = {}
        for uid in mutable:
            target = intents[uid]["target"]
            if target:
                targets.setdefault(target, []).append(uid)
        for target, contenders in targets.items():
            if len(contenders) > 1:
                winner = min(contenders)
                for uid in contenders:
                    if uid == winner:
                        continue
                    final_actions[uid] = self._wait_action()
                    intents[uid] = self._intent_from_action(final_actions[uid])
                    reviews.append(
                        {
                            "tick": self.tick,
                            "agent": uid,
                            "reason": "duplicate_claim",
                            "target": target,
                            "replacement": "Wait[]",
                        }
                    )

        # Existing peer claims and observed peer ownership veto only new proposals.
        for uid in sorted(mutable):
            spatial_error = self._spatial_action_error(final_actions[uid])
            if spatial_error is not None:
                final_actions[uid] = (None, None, spatial_error)
                intents[uid] = self._intent_from_action(final_actions[uid])
                reviews.append(
                    {
                        "tick": self.tick,
                        "agent": uid,
                        "reason": "spatial_action_guard",
                        "replacement": "replan",
                    }
                )
                continue

            signature = self._action_signature(final_actions[uid])
            if self.failure_counts.get(signature, 0) >= self.max_failed_action_retries:
                final_actions[uid] = (
                    None,
                    None,
                    "PeerConsult recovery check: this exact action already failed. "
                    "Change location, reference object, relation, or delegate to your "
                    "peer instead of repeating it.",
                )
                intents[uid] = self._intent_from_action(final_actions[uid])
                reviews.append(
                    {
                        "tick": self.tick,
                        "agent": uid,
                        "reason": "repeated_failed_action",
                        "replacement": "replan",
                    }
                )
                continue
            target = intents[uid]["target"]
            if target is None:
                continue
            owner = self.physical_owners.get(target)
            claimant = self.claims.get(target, {}).get("agent")
            reason: Optional[str] = None
            if owner is not None and owner != uid:
                reason = "ownership"
            elif claimant is not None and claimant != uid:
                reason = "claim_conflict"
            if reason is not None:
                final_actions[uid] = self._wait_action()
                intents[uid] = self._intent_from_action(final_actions[uid])
                reviews.append(
                    {
                        "tick": self.tick,
                        "agent": uid,
                        "reason": reason,
                        "target": target,
                        "replacement": "Wait[]",
                    }
                )

        for uid, intent in intents.items():
            self.current_intents[uid] = intent
            target = intent["target"]
            if uid in mutable and target is not None:
                self.claims[target] = {"agent": uid, "tick": self.tick}

        self.reviews.extend(reviews)
        return final_actions, reviews

    def record_execution_evidence(self, planner_info: Mapping[str, Any]) -> None:
        """Store skill feedback for recovery, without inferring task success."""
        for uid, ticket in planner_info.get("peerconsult_action_ticket", {}).items():
            key = (int(uid), int(ticket["id"]))
            # A skill can be observed as ongoing many times.  Preserve its first
            # record, then allow exactly the terminal observation to replace it.
            previous = self.execution_evidence.get(key)
            became_terminal = bool(ticket.get("terminal")) and not bool(
                previous and previous.get("terminal")
            )
            if previous is None or bool(ticket.get("terminal")):
                self.execution_evidence[key] = dict(ticket)
            if not became_terminal:
                continue
            response = str(ticket.get("response", ""))
            failed = any(
                marker in response.lower()
                for marker in (
                    "failed",
                    "no valid placement",
                    "not close",
                    "incorrect syntax",
                    "wrong use of api",
                )
            )
            self.last_terminal_feedback[int(uid)] = {
                "response": response,
                "failed": failed,
                "tick": self.tick,
                "action": ticket.get("action", ""),
                "input": ticket.get("input", ""),
            }
            if failed:
                action = (ticket.get("action"), ticket.get("input"), None)
                signature = self._action_signature(action)
                self.failure_counts[signature] = self.failure_counts.get(signature, 0) + 1

    def event(self, proposals: Mapping[int, Mapping[str, Any]], reviews: List[Dict[str, Any]], final_actions: Mapping[int, Action], planner_info: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "cards": {str(uid): self.decision_card(uid) for uid in proposals},
            "proposals": {
                str(uid): list(proposal["high_level_action"])
                for uid, proposal in proposals.items()
            },
            "reviews": reviews,
            "final_actions": {str(uid): list(action) for uid, action in final_actions.items()},
            "action_tickets": planner_info.get("peerconsult_action_ticket", {}),
            "verified_goal_delta": [],
        }
