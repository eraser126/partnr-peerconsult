#!/usr/bin/env python3
"""Apply two narrowly-scoped PARTNR Heuristic-Expert edge-case fixes safely."""

from pathlib import Path


ROOT = Path("/data/user/hd68631/projects/partnr-planner")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one exact match in {path}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Patched {path}")


replace_once(
    ROOT / "habitat_llm/world_model/world_graph.py",
    '''                elif furniture is None and (
                    (is_human_wg and self.agent_asymmetry)
                    or (not is_human_wg and self.world_model_type == "concept_graph")
                ):
                    # Objects are allowed to be marooned on unknown furniture under
                    # agent asymmetry condition, since the object may be placed anywhere
                    # in the house unbeknownst to the human agent
                    objs_info += obj.name + ": " + "unknown" + "\\n"
                else:
                    raise ValueError(f"Object {obj.name} has no parent")
''',
    '''                elif furniture is None:
                    # This method serializes the graph for planner logging. During
                    # simulator state updates an object can briefly have no graph
                    # parent (for example, after a placement). Preserve that fact in
                    # the log instead of aborting an otherwise valid evaluation.
                    # This does not mutate the graph or change planner actions.
                    objs_info += obj.name + ": " + "unknown" + "\\n"
''',
)

replace_once(
    ROOT / "habitat_llm/planner/scripted_centralized_planner.py",
    '''            objects_with_earlier_terminal = []
            if proposition.function_name != "is_next_to":
                objects_with_earlier_terminal = [
''',
    '''            objects_with_earlier_terminal = []
            # State predicates (clean, fill, power, ...) may intentionally act on
            # an object after an earlier terminal rearrangement proposition. Do not
            # remove those required objects from their candidate set.
            if (
                proposition.function_name not in OBJECT_STATES
                and proposition.function_name != "is_next_to"
            ):
                objects_with_earlier_terminal = [
''',
)
