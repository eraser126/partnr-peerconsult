"""Strict-observation, policy-light PeerConsult V4 board for PARTNR."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from habitat_llm.peer_consult.partnr_adapter import canonicalize_partnr_action

Action = Tuple[Optional[str], Optional[str], Optional[str]]


class PeerConsultBoard:
    """Centralizes facts and mutual exclusion, never a replacement policy."""

    PROTOCOL = "PeerConsultV4"
    _CLAIM_STAGES = {"acquire", "transport", "place", "state"}

    def __init__(self, claim_ttl_decisions=3, max_targets=5, max_rooms=4, max_reviews=2, max_execution_facts=3, placement_lock_ttl=12):
        self.claim_ttl_decisions = claim_ttl_decisions
        self.max_targets, self.max_rooms, self.max_reviews = max_targets, max_rooms, max_reviews
        self.max_execution_facts = max_execution_facts
        self.placement_lock_ttl = placement_lock_ttl
        self.reset()

    def reset(self) -> None:
        self.tick = self.protocol_step = 0
        self.physical_owners: Dict[str, int] = {}
        self.claims: Dict[str, Dict[str, Any]] = {}
        self.room_reservations: Dict[str, Dict[str, Any]] = {}
        self.settled_placements: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[int, Optional[str]] = {0: None, 1: None}
        self.active_executions: Dict[int, Dict[str, Any]] = {}
        self.current_intents: Dict[int, Dict[str, Any]] = {}
        self.execution_evidence: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._consumed_evidence: Dict[Tuple[int, int], bool] = {}
        self.execution_facts: Dict[int, List[str]] = {0: [], 1: []}
        self.required_recoveries: Dict[int, Optional[str]] = {0: None, 1: None}
        self.progress_versions, self.action_history = {0: 0, 1: 0}, {0: {}, 1: {}}
        self.pending_loop_guards: Dict[int, Optional[str]] = {0: None, 1: None}
        self.reviews: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    @staticmethod
    def _objects(graph: Any) -> Iterable[Any]:
        try:
            return graph.get_all_objects()
        except Exception:
            return []

    @staticmethod
    def _owner(graph: Any, obj: Any, uid: int) -> Optional[int]:
        try:
            if uid == 0 and graph.is_object_with_agent(obj, "robot"):
                return 0
            if uid == 1 and graph.is_object_with_agent(obj, "human"):
                return 1
        except Exception:
            pass
        return None

    @staticmethod
    def _wait() -> Action:
        return ("Wait", None, None)

    def _held(self, uid: int) -> List[str]:
        return sorted(name for name, owner in self.physical_owners.items() if owner == uid)

    def _event(self, uid: int, kind: str, subject: str) -> None:
        value = {"step": self.protocol_step, "agent": uid, "type": kind, "subject": subject}
        if value not in self.events:
            self.events.append(value)

    def _release(self, task_id: str) -> None:
        self.claims = {name: claim for name, claim in self.claims.items() if claim.get("task_id") != task_id}
        self.room_reservations = {room: claim for room, claim in self.room_reservations.items() if claim.get("task_id") != task_id}

    @staticmethod
    def _ticket_action(ticket: Mapping[str, Any]) -> str:
        action, value = str(ticket.get("action") or "Action"), ticket.get("input")
        return "{}[{}]".format(action, "" if value in {None, ""} else value)

    def _record_execution_fact(self, uid: int, ticket: Mapping[str, Any]) -> None:
        """Preserve bounded executor facts, never model thoughts or evaluator state."""
        action = self._ticket_action(ticket)
        response = " ".join(str(ticket.get("response") or "").split())[:160]
        success = ticket.get("outcome") == "terminal_success"
        fact = "{} {}{}".format(
            action,
            "succeeded" if success else "failed",
            ": {}".format(response) if response else "",
        )
        facts = self.execution_facts.setdefault(uid, [])
        facts.append(fact)
        self.execution_facts[uid] = facts[-self.max_execution_facts :]

        recovery = self.required_recoveries.get(uid)
        if success and recovery == action:
            self.required_recoveries[uid] = None

        action_name = str(ticket.get("action") or "").lower()
        action_input = str(ticket.get("input") or "")
        values = [value.strip() for value in action_input.split(",")]
        if success and action_name in {"place", "rearrange"} and len(values) >= 3:
            resource, relation, destination = values[:3]
            if resource and relation.lower() in {"on", "within"} and destination:
                self.settled_placements[resource] = {
                    "agent": uid,
                    "destination": destination,
                    "relation": relation.lower(),
                    "step": self.protocol_step,
                }
                self._event(uid, "placement_settled", "{}:{}:{}".format(resource, relation.lower(), destination))

        recovery_target = None
        if not success and "not close enough" in response.lower():
            if action_name == "pick" and action_input:
                recovery_target = action_input
            elif action_name in {"place", "rearrange"} and len(values) >= 3:
                recovery_target = values[2]
        if recovery_target:
            self.required_recoveries[uid] = "Navigate[{}]".format(recovery_target)

    def _consume_evidence(self) -> None:
        for key, ticket in list(self.execution_evidence.items()):
            if self._consumed_evidence.get(key) or not ticket.get("terminal"):
                continue
            self._consumed_evidence[key] = True
            task_id = ticket.get("task_id")
            task = self.tasks.get(task_id or "", {})
            uid = int(ticket.get("agent", task.get("agent", 0)))
            self._record_execution_fact(uid, ticket)
            if not task_id or task_id not in self.tasks:
                continue
            task, uid = self.tasks[task_id], int(self.tasks[task_id]["agent"])
            if ticket.get("outcome") == "terminal_success":
                task["status"] = "completed"
                self.progress_versions[uid] += 1
                self._event(uid, "task_completed", task_id)
            else:
                task["status"] = "suspended"
                self._event(uid, "task_failure", task_id)
            self._release(task_id)
            self.active_executions.pop(uid, None)
            if self.active_tasks.get(uid) == task_id:
                self.active_tasks[uid] = None

    def _expire(self) -> None:
        for name, claim in list(self.claims.items()):
            live = any(value.get("resource") == name for value in self.active_executions.values())
            if name in self.physical_owners or (not live and self.protocol_step - claim["step"] > self.claim_ttl_decisions):
                self.claims.pop(name, None)
                self._event(int(claim["agent"]), "claim_released", name)
        for room, claim in list(self.room_reservations.items()):
            live = any(value.get("room_scope") == room for value in self.active_executions.values())
            if not live and self.protocol_step - claim["step"] > self.claim_ttl_decisions:
                self.room_reservations.pop(room, None)
        for resource, placement in list(self.settled_placements.items()):
            if self.protocol_step - placement["step"] > self.placement_lock_ttl:
                self.settled_placements.pop(resource, None)

    @staticmethod
    def _matches_required_recovery(intent: Mapping[str, Any], recovery: Optional[str]) -> bool:
        if not recovery:
            return True
        action = str(intent.get("action", (None, None, None))[0] or "")
        value = str(intent.get("action", (None, None, None))[1] or "")
        return "{}[{}]".format(action, value) == recovery

    def observe(self, world_graphs: Mapping[int, Any]) -> None:
        self.tick += 1
        self._consume_evidence()
        prior, owners = dict(self.physical_owners), {}
        for uid, graph in world_graphs.items():
            for obj in self._objects(graph):
                name, owner = str(getattr(obj, "name", "")), self._owner(graph, obj, int(uid))
                if name and owner is not None:
                    owners[name] = owner
        self.physical_owners = owners
        for name, uid in owners.items():
            if prior.get(name) != uid:
                self.progress_versions[uid] += 1
                self._event(uid, "task_progress", "ownership:{}".format(name))
        self._expire()

    def _suspend_active(self, uid: int) -> None:
        task_id = self.active_tasks.get(uid)
        if task_id and self.tasks.get(task_id, {}).get("status") == "in_progress":
            self.tasks[task_id]["status"] = "suspended"
            self._release(task_id)
            self._event(uid, "task_suspended", task_id)
        self.active_executions.pop(uid, None)

    def _sync(self, intents: Mapping[int, Mapping[str, Any]], proposals: Mapping[int, Mapping[str, Any]]) -> None:
        for uid, intent in intents.items():
            task_id, stage = intent.get("task_id"), intent.get("stage")
            if not proposals[uid].get("is_new") or intent.get("error") or not task_id or stage in {"invalid", "wait", "done"}:
                continue
            if self.tasks.get(task_id, {}).get("status") == "completed":
                continue
            if self.active_tasks.get(uid) not in {None, task_id}:
                self._suspend_active(uid)
            task = self.tasks.setdefault(task_id, {"id": task_id, "agent": uid, "kind": stage, "status": "pending"})
            task.update(status="in_progress", action_identity=intent["action_identity"])
            self.active_tasks[uid] = task_id
            self._event(uid, "task_commitment", task_id)

    def _reject(self, uid, intent, final, reviews, reason) -> None:
        final[uid] = self._wait()
        task_id = intent.get("task_id")
        if task_id and self.tasks.get(task_id, {}).get("status") != "completed":
            self.tasks[task_id]["status"] = "suspended"
            self._release(task_id)
        if self.active_tasks.get(uid) == task_id:
            self.active_tasks[uid] = None
        self.active_executions.pop(uid, None)
        reviews.append({"step": self.protocol_step, "agent": uid, "reason": reason, "task_id": task_id, "verdict": "revise"})

    @staticmethod
    def _winner(resource: str, step: int, contenders: List[int]) -> int:
        digest = hashlib.sha256("{}:{}".format(step, resource).encode()).hexdigest()
        return sorted(contenders)[int(digest[:8], 16) % len(contenders)]

    def review(self, proposals: Mapping[int, Mapping[str, Any]]):
        """Atomically remove only illegal/conflicting proposals."""
        if any(value.get("is_new") for value in proposals.values()):
            self.protocol_step += 1
        final = {uid: tuple(value["high_level_action"]) for uid, value in proposals.items()}
        intents = {uid: dict(value.get("intent") or canonicalize_partnr_action(uid, final[uid])) for uid, value in proposals.items()}
        self._sync(intents, proposals)
        reviews: List[Dict[str, Any]] = []
        mutable = {uid for uid, value in proposals.items() if value.get("is_new") and not intents[uid].get("error") and intents[uid].get("stage") != "invalid"}
        for uid in sorted(mutable):
            intent, task_id = intents[uid], intents[uid].get("task_id")
            if intent.get("stage") == "done":
                self._reject(uid, intent, final, reviews, "completion_unavailable")
            elif self.tasks.get(task_id or "", {}).get("status") == "completed":
                self._reject(uid, intent, final, reviews, "completed_task")
            elif not self._matches_required_recovery(intent, self.required_recoveries.get(uid)):
                self._reject(uid, intent, final, reviews, "required_recovery")
            elif self.pending_loop_guards.get(uid) == intent.get("action_identity"):
                self.pending_loop_guards[uid] = None
                self._reject(uid, intent, final, reviews, "planning_loop_guard")
        grouped: Dict[str, List[int]] = {}
        for uid in mutable:
            if final[uid] == proposals[uid]["high_level_action"]:
                intent = intents[uid]
                if intent.get("stage") in self._CLAIM_STAGES and intent.get("resource"):
                    grouped.setdefault(str(intent["resource"]), []).append(uid)
        for resource, contenders in grouped.items():
            if len(contenders) > 1:
                winner = self._winner(resource, self.protocol_step, contenders)
                for uid in contenders:
                    if uid != winner:
                        self._reject(uid, intents[uid], final, reviews, "duplicate_claim")
        room_grouped: Dict[str, List[int]] = {}
        for uid in mutable:
            if final[uid] == proposals[uid]["high_level_action"]:
                room = intents[uid].get("room_scope")
                if room:
                    room_grouped.setdefault(str(room), []).append(uid)
        for room, contenders in room_grouped.items():
            if len(contenders) > 1:
                winner = self._winner("room:" + room, self.protocol_step, contenders)
                for uid in contenders:
                    if uid != winner:
                        self._reject(uid, intents[uid], final, reviews, "duplicate_room_reservation")
        for uid in sorted(mutable):
            if final[uid] != proposals[uid]["high_level_action"]:
                continue
            intent, resource = intents[uid], intents[uid].get("resource")
            settled = self.settled_placements.get(str(resource)) if resource else None
            if settled and intent.get("name") in {"pick", "rearrange"}:
                self._reject(uid, intent, final, reviews, "settled_relation")
                continue
            if intent.get("stage") in self._CLAIM_STAGES and resource:
                if intent.get("stage") in {"acquire", "transport"} and self._held(uid):
                    self._reject(uid, intent, final, reviews, "no_free_hand")
                    continue
                if self.physical_owners.get(str(resource)) not in {None, uid}:
                    self._reject(uid, intent, final, reviews, "physical_ownership")
                    continue
                if self.claims.get(str(resource), {}).get("agent") not in {None, uid}:
                    self._reject(uid, intent, final, reviews, "existing_claim")
                    continue
            room = intent.get("room_scope")
            if room and self.room_reservations.get(str(room), {}).get("agent") not in {None, uid}:
                self._reject(uid, intent, final, reviews, "room_reservation")
        for uid, intent in intents.items():
            self.current_intents[uid] = dict(intent)
            if final[uid] != proposals[uid]["high_level_action"] or not proposals[uid].get("is_new"):
                continue
            resource, task_id = intent.get("resource"), intent.get("task_id")
            if intent.get("stage") in self._CLAIM_STAGES and resource:
                self.claims[str(resource)] = {"agent": uid, "task_id": task_id, "step": self.protocol_step}
                self._event(uid, "claim_acquired", str(resource))
            if intent.get("room_scope"):
                self.room_reservations[str(intent["room_scope"])] = {"agent": uid, "task_id": task_id, "step": self.protocol_step}
            prior = self.action_history[uid]
            if prior.get("identity") == intent.get("action_identity") and prior.get("progress") == self.progress_versions[uid]:
                self.pending_loop_guards[uid] = intent.get("action_identity")
                if task_id in self.tasks:
                    self.tasks[task_id]["status"] = "suspended"
                    self._release(task_id)
            self.action_history[uid] = {"identity": intent.get("action_identity"), "progress": self.progress_versions[uid]}
            if task_id and intent.get("stage") not in {"wait", "done"}:
                self.active_executions[uid] = dict(intent)
        self.reviews.extend(reviews)
        return final, reviews, intents

    def _tasks(self, uid: int, status: str) -> List[str]:
        return sorted(task["id"] for task in self.tasks.values() if task.get("agent") == uid and task.get("status") == status)[-self.max_targets:]

    def decision_card(self, uid: int) -> str:
        """V2 supplies private entities via world_description; card supplies facts only."""
        peer = 1 - uid
        reviews = [item["reason"] for item in reversed(self.reviews) if item["agent"] == uid][:self.max_reviews]
        claims = sorted(name for name, value in self.claims.items() if value.get("agent") == peer)
        rooms = sorted(name for name, value in self.room_reservations.items() if value.get("agent") == peer)
        return "\n".join([
            "[PeerConsult V4 Decision Card]",
            "strict_observation=true; completion_oracle=unavailable_to_planner",
            "self_held={}".format(self._held(uid) or ["none"]),
            "self_active_tasks={}".format(self._tasks(uid, "in_progress") or ["none"]),
            "self_suspended_tasks={}".format(self._tasks(uid, "suspended") or ["none"]),
            "self_completed_tasks={}".format(self._tasks(uid, "completed") or ["none"]),
            "peer_public_held={}".format(self._held(peer) or ["none"]),
            "peer_public_tasks={}".format(self._tasks(peer, "in_progress") or ["none"]),
            "peer_claims={}".format(claims or ["none"]),
            "peer_room_reservations={}".format(rooms or ["none"]),
            "settled_placements={}".format(
                ["{}:{}:{}".format(resource, value["relation"], value["destination"])
                 for resource, value in sorted(self.settled_placements.items())][-self.max_targets:] or ["none"]
            ),
            "validator_feedback={}".format(reviews or ["none"]),
            "loop_guard={}".format(self.pending_loop_guards[uid] or "none"),
            "self_recent_execution={}".format(self.execution_facts[uid] or ["none"]),
            "self_required_recovery={}".format(self.required_recoveries[uid] or "none"),
            "Use exact entity IDs from your private world description; never add a label such as room: before an ID.",
        ])

    def record_execution_evidence(self, planner_info: Mapping[str, Any]) -> None:
        for uid, ticket in planner_info.get("peerconsult_action_ticket", {}).items():
            key = (int(uid), int(ticket["id"]))
            if key not in self.execution_evidence or ticket.get("terminal"):
                stored = dict(ticket)
                stored["agent"] = int(uid)
                self.execution_evidence[key] = stored

    def event(self, cards, proposals, reviews, final_actions, intents, planner_info):
        return {
            "protocol": self.PROTOCOL, "tick": self.tick,
            "cards": {str(uid): card for uid, card in cards.items()},
            "proposals": {str(uid): list(value["high_level_action"]) for uid, value in proposals.items()},
            "intents": {str(uid): dict(value) for uid, value in intents.items()},
            "validator": reviews,
            "final_actions": {str(uid): list(value) for uid, value in final_actions.items()},
            "action_tickets": planner_info.get("peerconsult_action_ticket", {}),
            "goal_completion": "unavailable_without_privileged_evaluator",
        }
