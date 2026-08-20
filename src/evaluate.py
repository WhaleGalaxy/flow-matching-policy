"""Rollout / evaluation: run a (EMA-averaged) policy in a ManiSkill3 env and
report success rate over N episodes.

Checklist ids: w1d4t2 (Euler rollout for FM), w2d0t0/t1 (sweep + stats table),
w2d3t0 (wrap into the ManiSkill3 BasePolicy submission format).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def rollout(policy, env, stats: dict, n_steps: int = 10) -> bool:
    """Run one episode with Euler integration of the FM velocity field.

    TODO (w1d4t2): preprocess obs -> policy.encode_obs -> integrate x from
    N(0, I) via `n_steps` Euler steps -> de-normalize with `stats` -> step env.
    Returns whether the episode ended in success (info['success']).
    """
    raise NotImplementedError("w1d4t2")


def evaluate(policy, envs: dict, n_episodes: int = 100, stats: dict | None = None) -> dict:
    """Run `rollout` n_episodes times per task, return {task: success_rate}.

    TODO (w2d0t0): loop over envs dict, average rollout() outcomes per task.
    """
    raise NotImplementedError("w2d0t0")
