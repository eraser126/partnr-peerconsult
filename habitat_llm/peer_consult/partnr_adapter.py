"""PARTNR adapters for the policy-light PeerConsult V4 core.

The adapter validates and canonicalizes an action selected by the local
planner.  It intentionally does not rank entities, choose alternatives, or
read evaluator propositions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Optional, Tuple


Action = Tuple[Optional[str], Optional[str], Optional[str]]
_TYPE_PREFIX = re.compile(
    r"^(?:room|object|furniture|receptacle|container|entity|target|reference)\s*:\s*(.+)$",
    re.IGNORECASE,
)


def _names(items: Iterable[Any]) -> set[str]:
    return {
        str(getattr(item, "name", "")).strip()
        for item in items
        if getattr(item, "name", "")
    }


def _world_names(world_graph: Any, method: str) -> set[str]:
    try:
        return _names(getattr(world_graph, method)())
    except Exception:
        return set()


def _object_furniture_name(world_graph: Any, object_name: str) -> Optional[str]:
    """Return a locally observed object's supporting furniture, if known."""
    if world_graph is None:
        return None
    try:
        for obj in world_graph.get_all_objects():
            if str(getattr(obj, "name", "")) == object_name:
                furniture = world_graph.find_furniture_for_object(obj)
                return str(getattr(furniture, "name", "")) or None
    except Exception:
        return None
    return None


def build_local_action_candidates(
    agent_uid: int, world_graph: Any, available_actions: Iterable[str]
) -> str:
    """Compile complete local action domains without choosing a policy.

    The result is deliberately a Cartesian-domain description instead of an
    exponentially long enumeration of every rearrangement.  It contains every
    entity currently legal in this agent's partial world graph and never uses
    peer-only or evaluator state.
    """
    tools = {str(action).lower() for action in available_actions}
    lines = [
        "[Complete local legal-action domains]",
        "Use exact IDs from your private world description above; do not invent IDs.",
        "This is a complete domain, not a priority order. Entity IDs are not repeated here.",
    ]
    if "explore" in tools:
        lines.append("Explore[room]: any locally known room ID.")
    if "navigate" in tools:
        lines.append("Navigate[target]: any locally known object, furniture, or room ID.")
    if "pick" in tools:
        lines.append("Pick[object]: any locally known object ID.")
    if "rearrange" in tools:
        lines.append(
            "Rearrange[object,relation,destination,constraint,reference]: "
            "object is any locally known object; relation in [on, within]; "
            "destination is any locally known furniture; constraint is None or "
            "next_to; reference is any locally known object when next_to."
        )
    if "place" in tools:
        lines.append(
            "Place[held_object,relation,destination,constraint,reference]: "
            "held_object must be shown as held by you; relation in [on, within]; "
            "destination is any locally known furniture; constraint is None or "
            "next_to; reference is any locally known object when next_to."
        )
    for action, target_type in (
        ("clean", "object"),
        ("fill", "object"),
        ("pour", "object"),
        ("poweron", "object"),
        ("poweroff", "object"),
        ("open", "furniture"),
        ("close", "furniture"),
    ):
        if action in tools:
            lines.append("{}[target]: any locally known {} ID.".format(action.title(), target_type))
    if "wait" in tools:
        lines.append("Wait[]: use only while an accepted skill is still executing.")
    lines.append("Done[] is runner-owned: do not emit it; the official completion adapter ends a solved episode.")
    return "\n".join(lines)


def _parts(value: Optional[str]) -> list[str]:
    return [part.strip() for part in str(value or "").split(",")]


def _none(value: str) -> bool:
    return value.strip().lower() in {"", "none", "null"}


def _canonical_value(value: str) -> tuple[str, bool]:
    match = _TYPE_PREFIX.match(value.strip())
    if not match:
        return value.strip(), False
    normalized = match.group(1).strip()
    return normalized, bool(normalized)


def _call(name: str, values: list[str]) -> str:
    return "{}[{}]".format(name, ",".join(values))


def canonicalize_partnr_action(
    agent_uid: int, action: Action, world_graph: Any = None
) -> Dict[str, Any]:
    """Return a stable, locally valid semantic intent for a PARTNR action."""
    action_name, action_input, parse_error = action
    name = str(action_name or "").strip()
    lowered = name.lower()
    values = []
    normalizations = []
    for raw in _parts(action_input):
        value, changed = _canonical_value(raw)
        values.append(value)
        if changed:
            normalizations.append({"from": raw, "to": value})
    canonical_action: Action = (
        name or None,
        ",".join(values) if action_input is not None else None,
        parse_error,
    )
    base: Dict[str, Any] = {
        "agent": int(agent_uid),
        "action": canonical_action,
        "name": lowered,
        "stage": "invalid",
        "resource": None,
        "room_scope": None,
        "task_id": None,
        "action_identity": "{}|{}".format(lowered or "invalid", _call(name, values)),
        "error": parse_error,
        "normalizations": normalizations,
    }
    if parse_error:
        return base

    objects = _world_names(world_graph, "get_all_objects") if world_graph is not None else set()
    rooms = _world_names(world_graph, "get_all_rooms") if world_graph is not None else set()
    furniture = (
        _world_names(world_graph, "get_all_furnitures")
        | _world_names(world_graph, "get_all_receptacles")
        if world_graph is not None
        else set()
    )

    def invalid(reason: str) -> Dict[str, Any]:
        base["error"] = reason
        return base

    if lowered in {"wait", "done"}:
        if values and not _none(values[0]):
            return invalid("{} takes no arguments.".format(name))
        base.update({"stage": lowered, "action_identity": "{}|{}".format(lowered, _call(name, []))})
        return base

    if lowered in {"rearrange", "place"}:
        if len(values) != 5 or any(_none(value) for value in values[:3]):
            return invalid("{} requires object, relation, furniture, constraint, reference.".format(name))
        obj, relation, destination, constraint, reference = values
        if relation.lower() not in {"on", "within"}:
            return invalid("{} relation must be on or within.".format(name))
        if constraint.lower() not in {"none", "next_to"}:
            return invalid("{} constraint must be None or next_to.".format(name))
        if constraint.lower() == "next_to" and _none(reference):
            return invalid("{} next_to requires a reference object.".format(name))
        if objects and obj not in objects:
            return invalid("{} source object is not locally known.".format(name))
        if furniture and destination not in furniture:
            return invalid("{} destination furniture is not locally known.".format(name))
        if constraint.lower() == "next_to" and objects and reference not in objects:
            return invalid("{} reference object is not locally known.".format(name))
        if constraint.lower() == "next_to":
            reference_furniture = _object_furniture_name(world_graph, reference)
            if reference_furniture is not None and reference_furniture != destination:
                return invalid(
                    "{} next_to reference must already be on destination {} (it is on {}).".format(
                        name, destination, reference_furniture
                    )
                )
        stage = "transport" if lowered == "rearrange" else "place"
        base.update(
            {
                "stage": stage,
                "resource": obj,
                # A delivery stays the same task whichever agent performs it.
                "task_id": "delivery:{}:{}:{}:{}:{}".format(
                    obj, relation.lower(), destination, constraint.lower(), reference or "none"
                ),
                "action_identity": "{}|{}".format(stage, _call(name, values)),
            }
        )
        return base

    if lowered in {"pick", "clean", "fill", "pour", "open", "close", "poweron", "poweroff"}:
        if len(values) != 1 or _none(values[0]):
            return invalid("{} requires exactly one target.".format(name))
        target = values[0]
        if lowered in {"pick", "clean", "fill", "poweron", "poweroff"} and objects and target not in objects:
            return invalid("{} target is not locally known.".format(name))
        if lowered in {"open", "close"} and furniture and target not in furniture:
            return invalid("{} target is not locally known.".format(name))
        stage = "acquire" if lowered == "pick" else "state"
        base.update(
            {
                "stage": stage,
                "resource": target,
                "task_id": "entity:{}".format(target) if stage == "acquire" else "state:{}:{}".format(lowered, target),
                "action_identity": "{}|{}".format(stage, _call(name, values)),
            }
        )
        return base

    if lowered == "explore":
        if len(values) != 1 or _none(values[0]):
            return invalid("Explore requires exactly one room.")
        room = values[0]
        if rooms and room not in rooms:
            return invalid("Explore room is not locally known.")
        base.update(
            {
                "stage": "explore",
                "resource": room,
                "room_scope": room,
                "task_id": "room:{}".format(room),
                "action_identity": "explore|{}".format(_call(name, values)),
            }
        )
        return base

    if lowered == "navigate":
        if len(values) != 1 or _none(values[0]):
            return invalid("Navigate requires exactly one target.")
        target = values[0]
        known = objects | rooms | furniture
        if known and target not in known:
            return invalid("Navigate target is not locally known.")
        if target in rooms:
            base.update(
                {
                    # Navigating to a known room is not blind exploration.
                    # It must not reserve or deduplicate that room scope.
                    "stage": "navigate",
                    "resource": target,
                    "task_id": "navigate:room:{}".format(target),
                    "action_identity": "navigate-room|{}".format(_call(name, values)),
                }
            )
        else:
            base.update(
                {
                    "stage": "navigate",
                    "resource": target,
                    "task_id": "navigate:{}".format(target),
                    "action_identity": "navigate|{}".format(_call(name, values)),
                }
            )
        return base

    return invalid("Unsupported PARTNR action for PeerConsult V4.")


def execution_outcome(response: Any, terminal: bool) -> str:
    """Prefer structured executor status, with compatibility for legacy text."""
    if not terminal:
        return "ongoing"
    if isinstance(response, Mapping):
        status = str(response.get("status", response.get("outcome", ""))).lower()
        if status in {"success", "succeeded", "completed", "terminal_success"}:
            return "terminal_success"
        if status in {"failure", "failed", "error", "terminal_failure"}:
            return "terminal_failure"
        if response.get("success") is True:
            return "terminal_success"
        if response.get("success") is False:
            return "terminal_failure"
    value = str(response or "").strip().lower()
    if value in {"successful execution!", "success", "succeeded"} or value.endswith(" was a success"):
        return "terminal_success"
    return "terminal_failure"
