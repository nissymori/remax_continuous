"""SAC for continuous actions, as in rejax, with replay minibatches drawn with replacement."""

import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax import numpy as jnp
from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch
from rejax.networks import QNetwork, SquashedGaussianPolicy

from common import METRICS, entropy_terms, log_metrics, sample_minibatch


class SAC(
    ReplayBufferMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    TargetNetworkMixin,
    Algorithm,
):
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_critics: int = struct.field(pytree_node=False, default=2)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    target_entropy_ratio: float = struct.field(pytree_node=True, default=0.98)  # discrete actions only
    log_interval: int = struct.field(pytree_node=False, default=0)
    log_callback: callable = struct.field(pytree_node=False, default=None)

    soft_target = True  # entropy term in the critic target, and temperature tuning

    def make_act(self, ts):
        def act(obs, rng):
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)
            action = self.actor.apply(ts.actor_ts.params, jnp.expand_dims(obs, 0), rng, method="act")
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        agent_kwargs = config.pop("agent_kwargs", {})
        agent_kwargs["activation"] = getattr(nn, agent_kwargs.pop("activation", "relu"))
        agent_kwargs["hidden_layer_sizes"] = tuple(config.pop("hidden_layer_sizes", (64, 64)))
        action_space = env.action_space(env_params)
        actor = SquashedGaussianPolicy(
            np.prod(action_space.shape),
            (action_space.low, action_space.high),
            log_std_range=(-10, 2),
            **agent_kwargs,
        )
        return {"actor": actor, "critic": QNetwork(**agent_kwargs)}

    @property
    def target_entropy(self):
        return -self.action_dim

    @property
    def vmap_critic(self):
        return jax.vmap(self.critic.apply, in_axes=(0, None, None))

    def actor_tx(self):
        return optax.adam(learning_rate=self.learning_rate)

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.empty((1, *self.obs_space.shape))
        act_ph = jnp.empty((1, *self.env.action_space(self.env_params).shape))
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        actor_params = self.actor.init(rng_actor, obs_ph, rng_actor)
        rng_critic = jax.random.split(rng_critic, self.num_critics)
        critic_params = jax.vmap(self.critic.init, in_axes=(0, None, None))(rng_critic, obs_ph, act_ph)

        clip = optax.clip(self.max_grad_norm)
        tx = optax.chain(clip, optax.adam(learning_rate=self.learning_rate))
        state = {
            "actor_ts": TrainState.create(
                apply_fn=(), params=actor_params, tx=optax.chain(clip, self.actor_tx())
            ),
            "critic_ts": TrainState.create(apply_fn=(), params=critic_params, tx=tx),
            "critic_target_params": critic_params,
        }
        if self.soft_target:
            alpha_params = FrozenDict({"log_alpha": jnp.array(0.0)})
            state["alpha_ts"] = TrainState.create(apply_fn=(), params=alpha_params, tx=tx)
        return state

    def train_iteration(self, ts):
        old_global_step = ts.global_step
        ts, batch = self.collect_transitions(ts)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

        def update_iteration(_, carry):
            ts, metrics_sum = carry
            rng, rng_sample = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            mb = sample_minibatch(ts.replay_buffer, self.batch_size, rng_sample)
            if self.normalize_observations:
                mb = mb._replace(
                    obs=self.normalize_obs(ts.obs_rms_state, mb.obs),
                    next_obs=self.normalize_obs(ts.obs_rms_state, mb.next_obs),
                )
            if self.normalize_rewards:
                mb = mb._replace(reward=self.normalize_rew(ts.rew_rms_state, mb.reward))
            ts, metrics = self.update(ts, mb)
            return ts, jax.tree.map(jnp.add, metrics_sum, metrics)

        def do_updates(ts):
            zeros = {k: jnp.array(0.0) for k in METRICS}
            ts, metrics_sum = jax.lax.fori_loop(0, self.num_epochs, update_iteration, (ts, zeros))
            return ts, jax.tree.map(lambda x: x / self.num_epochs, metrics_sum)

        start_training = ts.global_step > self.fill_buffer
        ts, metrics = jax.lax.cond(
            start_training, do_updates, lambda ts: (ts, {k: jnp.array(0.0) for k in METRICS}), ts
        )
        target_params = self.maybe_update_target_params(
            ts.critic_ts.params, ts.critic_target_params, ts.global_step, old_global_step
        )
        ts = ts.replace(critic_target_params=target_params)
        log_metrics(self.log_callback, self.log_interval, ts.global_step, start_training, metrics)
        return ts

    def collect_transitions(self, ts):
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        last_obs = ts.last_obs
        if self.normalize_observations:
            last_obs = self.normalize_obs(ts.obs_rms_state, last_obs)
        actions = self.actor.apply(ts.actor_ts.params, last_obs, rng_action, method="act")

        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(
            rng_steps, ts.env_state, actions, self.env_params
        )
        if self.normalize_observations:
            ts = ts.replace(obs_rms_state=self.update_obs_rms(ts.obs_rms_state, next_obs))
        if self.normalize_rewards:
            ts = ts.replace(rew_rms_state=self.update_rew_rms(ts.rew_rms_state, rewards, dones))

        batch = Minibatch(obs=ts.last_obs, action=actions, reward=rewards, next_obs=next_obs, done=dones)
        ts = ts.replace(last_obs=next_obs, env_state=env_state, global_step=ts.global_step + self.num_envs)
        return ts, batch

    def actor_loss(self, ts, params, obs, action_rng):
        """-> (loss, log-probabilities of the sampled actions)."""
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])
        action, logprob = self.actor.apply(params, obs, action_rng, method="action_log_prob")
        qs = self.vmap_critic(ts.critic_ts.params, obs, action)
        return (alpha * logprob - qs.min(axis=0)).mean(), logprob

    def update_actor(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        params = ts.actor_ts.params
        (loss, logprob), grads = jax.value_and_grad(
            lambda p: self.actor_loss(ts, p, mb.obs, action_rng), has_aux=True
        )(params)
        actor_ts = ts.actor_ts.apply_gradients(grads=grads)
        log_std_sum, tanh_log_det = entropy_terms(self.actor, params, mb.obs, action_rng)
        metrics = {
            "actor_loss": loss,
            "entropy": (-logprob).mean(),
            "actor_grad_norm": optax.global_norm(grads),
            "actor_update_norm": optax.global_norm(jax.tree.map(jnp.subtract, actor_ts.params, params)),
            "log_std_sum": log_std_sum,
            "tanh_log_det": tanh_log_det,
        }
        return ts.replace(actor_ts=actor_ts), logprob, metrics

    def update_critic(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def loss_fn(params):
            action, logprob = self.actor.apply(
                ts.actor_ts.params, mb.next_obs, action_rng, method="action_log_prob"
            )
            q_target = jnp.min(self.vmap_critic(ts.critic_target_params, mb.next_obs, action), axis=0)
            if self.soft_target:
                q_target = q_target - jnp.exp(ts.alpha_ts.params["log_alpha"]) * logprob
            target = mb.reward + self.gamma * (1 - mb.done) * q_target
            qs = self.vmap_critic(params, mb.obs, mb.action)
            return jax.vmap(lambda q: optax.l2_loss(q, target))(qs).sum(axis=0).mean()

        loss, grads = jax.value_and_grad(loss_fn)(ts.critic_ts.params)
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads)), loss

    def update_alpha(self, ts, logprob):
        def loss_fn(params):
            return (-jnp.exp(params["log_alpha"]) * (logprob + self.target_entropy)).mean()

        grads = jax.grad(loss_fn)(ts.alpha_ts.params)
        return ts.replace(alpha_ts=ts.alpha_ts.apply_gradients(grads=grads))

    def update(self, ts, mb):
        ts, logprob, metrics = self.update_actor(ts, mb)
        ts, metrics["critic_loss"] = self.update_critic(ts, mb)
        if self.soft_target:
            ts = self.update_alpha(ts, logprob)
        return ts, metrics
