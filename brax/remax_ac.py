"""ReMAC: SAC's actor-critic with the ReMax objective as the actor loss and no entropy term."""

import jax
import optax
from flax import struct
from jax import numpy as jnp

from common import remax
from sac import SAC


class ReMaxAC(SAC):
    remax_m: int = struct.field(pytree_node=False, default=4)  # retry budget M
    remax_num_samples: int = struct.field(pytree_node=False, default=8)  # action samples B per state
    actor_epsilon: float = struct.field(pytree_node=False, default=1e-8)  # Adam's epsilon (actor)
    actor_optimizer: str = struct.field(pytree_node=False, default="adam")  # "adam" or "sgd"

    soft_target = False

    def actor_tx(self):
        if self.actor_optimizer == "adam":
            return optax.adam(learning_rate=self.learning_rate, eps=float(self.actor_epsilon))
        if self.actor_optimizer == "sgd":
            return optax.sgd(learning_rate=self.learning_rate, momentum=0.0)
        raise ValueError(f"Unknown actor optimizer '{self.actor_optimizer}'.")

    def actor_loss(self, ts, params, obs, action_rng):
        """-> (-mean over states of rho^M of the Q-values of B sampled actions, log-probabilities)."""
        b = self.remax_num_samples
        keys = jax.random.split(action_rng, b)
        actions = jax.vmap(lambda k: self.actor.apply(params, obs, k, method="action_log_prob")[0])(keys)
        actions = jnp.swapaxes(actions, 0, 1)  # (states, B, action_dim)
        obs_b = jnp.repeat(obs[:, None, :], b, axis=1)
        qs = self.vmap_critic(
            ts.critic_ts.params, obs_b.reshape(-1, obs.shape[-1]), actions.reshape(-1, actions.shape[-1])
        )
        q = jnp.min(qs, axis=0).reshape(obs.shape[0], b)
        _, logprob = self.actor.apply(params, obs, action_rng, method="action_log_prob")
        return -jax.vmap(remax, in_axes=(0, None))(q, self.remax_m).mean(), logprob
