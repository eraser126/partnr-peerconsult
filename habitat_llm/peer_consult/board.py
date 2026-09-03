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

    def __init__(
        self,
        claim_ttl_decisions=3,
        max_targets=5,
        max_rooms=4,
        max_reviews=2,
        max_execution_facts=3,
        placement_lock_ttl=12,
        max_ledger_entries=8,
        max_public_report_chars=480,
        publish_exploration_reports=False,
        enforce_room_exploration_dedup=False,
        enforce_required_recovery=False,
        enforce_placement_stability=False,
        share_discovered_locations=False,
        max_shared_locations=5,
    ):
        self.claim_ttl_decisions = claim_ttl_decisions
        self.max_targets, self.max_rooms, self.max_reviews = max_targets, max_rooms, max_reviews
        self.max_execution_facts = max_execution_facts
        self.placement_lock_ttl = placement_lock_ttl
        self.max_ledger_entries = max_ledger_entries
        self.max_public_report_chars = max_public_report_chars
        # These are optional PARTNR research aids, not V4 core rules.  Core
        # coordination may reject only factual conflicts and safety violations.
        self.publish_exploration_reports = publish_exploration_reports
        self.enforce_room_exploration_dedup = enforce_room_exploration_dedup
        self.enforce_required_recovery = enforce_required_recovery
        self.enforce_placement_stability = enforce_placement_stability
        self.share_discovered_locations = share_discovered_locations
        self.max_shared_locations = max_shared_locations
        self.reset()

    def reset(self) -> None:
        self.tick = self.protocol_step = 0
        self.physical_owners: Dict[str, int] = {}
        self.claims: Dict[str, Dict[str, Any]] = {}
        self.room_reservations: Dict[str, Dict[str, Any]] = {}
        self.settled_placements: Dict[str, Dict[str, Any]] = {}
        # These records are built only from public actions and their executor
        # replies.  They are durable task memory, not evaluator propositions.
        self.known_rooms: Dict[int, set[str]] = {0: set(), 1: set()}
        self.explored_rooms: Dict[str, Dict[str, Any]] = {}
        self.object_ledger: Dict[str, Dict[str, Any]] = {}
        self.discovered_locations: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[int, Optional[str]] = {0: None, 1: None}
        self.active_executions: Dict[int, Dict[str, Any]] = {}
        self.current_intents: Dict[int, Dict[str, Any]] = {}
        self.execution_evidence: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._consumed_evidence: Dict[Tuple[int, int], bool] = {}
        self.execution_facts: Dict[int, List[str]] = {0: [], 1: []}
        self.required_recoveries: Dict[int, Optional[str]] = {0: None, 1: None}
        self.recovery_advice: Dict[int, Optional[str]] = {0: None, 1: None}
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
    def _rooms(graph: Any) -> Iterable[Any]:
        try:
            return graph.get_all_rooms()
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

    @staticmethod
    def _location(graph: Any, obj: Any) -> Tuple[Optional[str], Optional[str]]:
        """Return facts present in one agent's partial graph, if any."""
        try:
            furniture = graph.find_furniture_for_object(obj)
            furniture_name = str(getattr(furniture, "name", "")) or None
        except Exception:
            furniture_name = None
        try:
            room = graph.get_room_for_entity(obj)
            room_name = str(getattr(room, "name", "")) or None
        except Exception:
            room_name = None
        return furniture_name, room_name

    def _remember_object(self, resource: str, uid: int, state: str, detail: str) -> None:
        if not resource:
            return
        self.object_ledger[resource] = {
            "agent": uid,
            "state": state,
            "detail": detail,
            "step": self.protocol_step,
        }

    @staticmethod
    def _public_room_report(response: str, limit: int) -> str:
        """Return a bounded report that the acting agent exposed via Explore."""
        raw = str(response or "")
        marker = raw.lower().find("objects:")
        if marker < 0:
            return "executor confirmed exploration"
        lines = [" ".join(line.split()) for line in raw[marker + len("objects:") :].splitlines()]
        report = "; ".join(line for line in lines if line)
        return (report or "no objects reported")[:limit]

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
        raw_response = str(ticket.get("response") or "")
        response = " ".join(raw_response.split())[:160]
        success = ticket.get("outcome") == "terminal_success"
        action_name = str(ticket.get("action") or "").lower()
        action_input = str(ticket.get("input") or "")
        fact = "{} {}{}".format(
            action,
            "succeeded" if success else "failed",
            ": {}".format(response) if response else "",
        )
        # Validator-inserted Wait actions are not physical experience.  Do not
        # let them crowd out the agent's useful action/failure trajectory.
        if not (action_name == "wait" and raw_response.startswith("validator:")):
            facts = self.execution_facts.setdefault(uid, [])
            facts.append(fact)
            self.execution_facts[uid] = facts[-self.max_execution_facts :]

        recovery = self.required_recoveries.get(uid)
        if success and recovery == action:
            self.required_recoveries[uid] = None
        if success and self.recovery_advice.get(uid) == action:
            self.recovery_advice[uid] = None

        values = [value.strip() for value in action_input.split(",")]
        if success and action_name == "explore" and action_input:
            self.explored_rooms[action_input] = {
                "agent": uid,
                "step": self.protocol_step,
                # Completion of the Explore task is a public lifecycle fact.
                # Its raw observation remains private unless an experiment
                # explicitly opts into publication.
                "report": (
                    self._public_room_report(raw_response, self.max_public_report_chars)
                    if self.publish_exploration_reports
                    else None
                ),
            }
            self._event(uid, "room_explored", action_input)
        if success and action_name == "pick" and action_input:
            self._remember_object(action_input, uid, "held", "Pick succeeded")
        elif success and action_name in {"place", "rearrange"} and len(values) >= 3:
            resource, relation, destination = values[:3]
            self._remember_object(
                resource,
                uid,
                "placed",
                "{} {}".format(relation.lower(), destination),
            )
        # Failures stay in `execution_facts[uid]`: they are useful private
        # recovery context, but are not durable public object facts.
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
            elif action_name == "rearrange" and values and values[0]:
                # Rearrange begins by acquiring its source object.  If that
                # acquisition fails, navigating to the destination repeats
                # the same precondition error.
                recovery_target = values[0]
            elif action_name == "place" and len(values) >= 3:
                recovery_target = values[2]
        if recovery_target:
            recovery = "Navigate[{}]".format(recovery_target)
            self.recovery_advice[uid] = recovery
            if self.enforce_required_recovery:
                self.required_recoveries[uid] = recovery

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

    def required_recovery_action(self, uid: int) -> Optional[Action]:
        """Expose one executor-derived recovery action, if any.

        This contains no evaluator state or target ranking: it is exactly the
        Navigate action derived from the same agent's failed skill response.
        """
        recovery = self.required_recoveries.get(uid)
        if not recovery or "[" not in recovery or not recovery.endswith("]"):
            return None
        name, value = recovery.split("[", 1)
        return (name or None, value[:-1].strip() or None, None)

    def observe(self, world_graphs: Mapping[int, Any]) -> None:
        self.tick += 1
        self._consume_evidence()
        prior, owners = dict(self.physical_owners), {}
        for uid, graph in world_graphs.items():
            self.known_rooms.setdefault(int(uid), set()).update(
                str(getattr(room, "name", "")) for room in self._rooms(graph) if getattr(room, "name", "")
            )
            for obj in self._objects(graph):
                name, owner = str(getattr(obj, "name", "")), self._owner(graph, obj, int(uid))
                if name and owner is not None:
                    owners[name] = owner
                if name and self.share_discovered_locations:
                    furniture, room = self._location(graph, obj)
                    if furniture or room:
                        self.discovered_locations[name] = {
                            "agent": int(uid), "furniture": furniture,
                            "room": room, "step": self.protocol_step,
                        }
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
            elif (
                self.enforce_required_recovery
                and not self._matches_required_recovery(intent, self.required_recoveries.get(uid))
            ):
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
            if (
                self.enforce_placement_stability
                and settled
                and intent.get("name") in {"pick", "rearrange"}
            ):
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
                continue
            explored = self.explored_rooms.get(str(room)) if room else None
            if (
                self.enforce_room_exploration_dedup
                and explored
                and explored.get("agent") != uid
            ):
                self._reject(uid, intent, final, reviews, "room_already_reported")
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

    def _completed_action_calls(self, uid: int) -> List[str]:
        """Return bounded, factual action calls already completed by this agent."""
        calls = {
            str(task.get("action_identity"))
            for task in self.tasks.values()
            if task.get("agent") == uid
            and task.get("status") == "completed"
            and task.get("action_identity")
        }
        return sorted(calls)[-self.max_targets:]

    def _available_rooms(self, uid: int) -> List[str]:
        reserved_by_peer = {
            room
            for room, value in self.room_reservations.items()
            if value.get("agent") != uid
        }
        return sorted(
            room
            for room in self.known_rooms.get(uid, set())
            if room not in self.explored_rooms and room not in reserved_by_peer
        )[: self.max_rooms]

    def _room_reports(self) -> List[str]:
        if not self.publish_exploration_reports:
            return []
        entries = []
        for room, value in sorted(self.explored_rooms.items(), key=lambda item: item[1]["step"], reverse=True):
            if value.get("report"):
                entries.append("{} by agent{}: {}".format(room, value["agent"], value["report"]))
        return entries[: self.max_rooms]

    def progress_signature(self) -> tuple:
        """Return monotonic, public task facts for the no-progress guard.

        This deliberately excludes raw world-graph churn and failed actions.
        It changes only after an accepted task produces a positive public fact.
        """
        completed = tuple(sorted(task_id for task_id, task in self.tasks.items() if task.get("status") == "completed"))
        # A current owner map is not monotonic (placing an object removes an
        # owner), so use the positive ownership/completion version instead.
        progress_versions = tuple(sorted(self.progress_versions.items()))
        explored = tuple(sorted(self.explored_rooms))
        ledger = tuple(
            sorted(
                (resource, value["state"], value["detail"])
                for resource, value in self.object_ledger.items()
                if value.get("state") in {"held", "placed"}
            )
        )
        return completed, progress_versions, explored, ledger

    def _ledger(self) -> List[str]:
        entries = []
        for resource, value in sorted(self.object_ledger.items(), key=lambda item: item[1]["step"], reverse=True):
            entries.append(
                "{}: {} by agent{} ({})".format(
                    resource, value["state"], value["agent"], value["detail"]
                )
            )
        return entries[: self.max_ledger_entries]

    def _shared_locations(self, uid: int) -> List[str]:
        if not self.share_discovered_locations:
            return []
        entries = []
        for resource, value in sorted(
            self.discovered_locations.items(),
            key=lambda item: item[1]["step"],
            reverse=True,
        ):
            if value.get("agent") == uid:
                continue
            location = value.get("furniture") or "unknown furniture"
            room = value.get("room") or "unknown room"
            entries.append("{} at {} in {} (agent{})".format(resource, location, room, value["agent"]))
        return entries[: self.max_shared_locations]

    def _revision_advice(self, uid: int) -> str:
        latest = next((item for item in reversed(self.reviews) if item["agent"] == uid), None)
        if not latest:
            return "none"
        reason = latest["reason"]
        guidance = {
            "duplicate_room_reservation": "Do not repeat that room action. Choose self_available_unexplored_rooms or use a reported object.",
            "room_reservation": "The peer owns that room now. Choose a different self_available_unexplored_room or an object action.",
            "room_already_reported": "That room has already been explored. Use public_room_reports instead of exploring it again.",
            "duplicate_claim": "The peer is working on that object. Select a different unclaimed object or room.",
            "physical_ownership": "The peer holds that object. Do not pick or rearrange it; work on another object.",
            "required_recovery": "Execute self_required_recovery exactly before proposing another action.",
            "completed_task": "That exact action already succeeded. Advance to a different unsatisfied object or room.",
            "planning_loop_guard": "The last identical action made no progress. Select a materially different action.",
        }
        return guidance.get(reason, "The previous proposal was rejected; choose a materially different valid action.")

    def decision_card(self, uid: int) -> str:
        """V2 supplies private entities via world_description; card supplies facts only."""
        peer = 1 - uid
        reviews = [item["reason"] for item in reversed(self.reviews) if item["agent"] == uid][:self.max_reviews]
        claims = sorted(name for name, value in self.claims.items() if value.get("agent") == peer)
        rooms = sorted(name for name, value in self.room_reservations.items() if value.get("agent") == peer)
        return "\n".join([
            "[PeerConsult V4 Decision Card]",
            "strict_observation=true; completion_status=not_supplied_to_planner",
            "self_held={}".format(self._held(uid) or ["none"]),
            "self_active_tasks={}".format(self._tasks(uid, "in_progress") or ["none"]),
            "self_suspended_tasks={}".format(self._tasks(uid, "suspended") or ["none"]),
            "self_completed_tasks={}".format(self._tasks(uid, "completed") or ["none"]),
            "self_completed_action_calls={} (never repeat an exact listed call)".format(
                self._completed_action_calls(uid) or ["none"]
            ),
            "peer_public_held={}".format(self._held(peer) or ["none"]),
            "peer_public_tasks={}".format(self._tasks(peer, "in_progress") or ["none"]),
            "peer_claims={}".format(claims or ["none"]),
            "peer_room_reservations={}".format(rooms or ["none"]),
            "self_available_unexplored_rooms={}".format(self._available_rooms(uid) or ["none"]),
            "public_room_reports={}".format(self._room_reports() or ["none"]),
            "public_object_ledger={}".format(self._ledger() or ["none"]),
            "peer_discovered_locations={}".format(self._shared_locations(uid) or ["disabled"]),
            "settled_placements={}".format(
                ["{}:{}:{}".format(resource, value["relation"], value["destination"])
                 for resource, value in sorted(self.settled_placements.items())][-self.max_targets:] or ["none"]
            ),
            "validator_feedback={}".format(reviews or ["none"]),
            "loop_guard={}".format(self.pending_loop_guards[uid] or "none"),
            "self_recent_execution={}".format(self.execution_facts[uid] or ["none"]),
            "self_required_recovery={}".format(self.required_recoveries[uid] or "none"),
            "self_recovery_advice={}".format(self.recovery_advice[uid] or "none"),
            "self_action_guidance={}".format(self._revision_advice(uid)),
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
