"""Sparse-reward Reacher: Brax's Reacher with the reward 1{d_t < 0.01}, d_t the distance between
the fingertip and the target.  Importing this module registers it as brax/sparse_reacher."""

import jax
import jax.numpy as jp
from brax import math
from brax.envs import register_environment
from brax.envs.base import State
from brax.envs.reacher import Reacher


class SparseReacher(Reacher):
    goal_radius: float = 0.01

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)
        # the fingertip-to-target vector is the last three entries of the observation
        dist = math.safe_norm(state.obs[-3:])
        return state.replace(reward=jp.where(dist < self.goal_radius, 1.0, 0.0))


register_environment("sparse_reacher", SparseReacher)
