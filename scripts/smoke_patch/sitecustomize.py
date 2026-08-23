"""Bound the invalid-navmesh fallback during the PARTNR smoke verifier only.

This module is loaded automatically through PYTHONPATH by partnr_smoke.sbatch.
It does not modify Habitat-Lab on disk and is not used by normal PARTNR runs.
For a valid navmesh, `safe_snap_point` returns immediately and behavior is
identical to Habitat-Lab.  If no point exists on the selected navmesh island,
the upstream fallback can make thousands of failed native sampling calls.  The
bounded version raises promptly, so PARTNR's verifier records that episode as
an initialization failure and moves on to the next episode.
"""

import os

import numpy as np

from habitat.tasks.rearrange.rearrange_sim import RearrangeSim


def _bounded_safe_snap_point(self, pos: np.ndarray) -> np.ndarray:
    new_pos = self.pathfinder.snap_point(pos, self._largest_indoor_island_idx)
    if not np.isnan(new_pos[0]):
        return np.array(new_pos)

    max_iter = int(os.environ.get("PARTNR_SMOKE_SNAP_MAX_ITER", "2"))
    num_sample_points = int(os.environ.get("PARTNR_SMOKE_SNAP_NUM_SAMPLES", "5"))
    offset_distance = 1.5
    distance_per_iter = 0.5
    for regen_i in range(max_iter):
        new_pos = self.pathfinder.get_random_navigable_point_near(
            pos,
            offset_distance + regen_i * distance_per_iter,
            num_sample_points,
            island_index=self._largest_indoor_island_idx,
        )
        if not np.isnan(new_pos[0]):
            return np.array(new_pos)

    raise RuntimeError(
        "No valid navigable start point after bounded smoke-test fallback: "
        f"scene_id={self.ep_info.scene_id}, episode_id={self.ep_info.episode_id}"
    )


RearrangeSim.safe_snap_point = _bounded_safe_snap_point
