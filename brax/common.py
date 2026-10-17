"""The ReMax estimator, replay sampling, and the training metrics shared by SAC and ReMAC."""

import distrax
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

METRICS = ("actor_loss", "critic_loss", "entropy", "actor_grad_norm", "actor_update_norm",
           "log_std_sum", "tanh_log_det")


def comb(n, k):
    """Binomial coefficient C(n, k), elementwise; 0 where n < k."""
    n = jnp.asarray(n, dtype=jnp.float32)
    k = jnp.asarray(k, dtype=jnp.float32)
    valid = n >= k
    n, k = jnp.where(valid, n, 0.0), jnp.where(valid, k, 0.0)
    return jnp.where(valid, jnp.exp(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)), 0.0)


def remax(q, m):
    """Unbiased estimate of E[max of m draws] from the B draws in q:
    C(B, m)^{-1} sum_i C(i - 1, m - 1) q_(i), with q sorted in ascending order."""
    b = q.shape[0]
    if b < m:
        raise ValueError(f"Batch size {b} must be greater than or equal to {m}.")
    return jnp.sum(comb(jnp.arange(b), m - 1) / comb(b, m) * jnp.sort(q))


def sample_minibatch(buffer, num, rng):
    """`num` transitions drawn uniformly, with replacement, from the filled part of `buffer`."""
    idx = jax.random.randint(rng, (num,), 0, buffer.num_entries)
    return jax.tree.map(lambda arr: arr[idx], buffer.data)


def entropy_terms(actor, params, obs, rng):
    """The two terms of the entropy of the tanh-squashed Gaussian policy (beyond d/2 log 2 pi e):
    sum_i log sigma_i and E[sum_i log(1 - tanh(u_i)^2)], averaged over the states in `obs`."""
    dist = actor.apply(params, obs, method="_action_dist")
    u = dist.sample(seed=rng)
    log_det = distrax.Tanh().forward_log_det_jacobian(u).sum(axis=-1)
    return jnp.mean(jnp.log(dist.stddev()).sum(axis=-1)), jnp.mean(log_det)


def log_metrics(callback, interval, step, is_training, metrics):
    """Send `metrics` to `callback(step, seed, metrics)` every `interval` steps of training."""
    if callback is None or interval <= 0:
        return
    seed = jax.lax.axis_index("seed")
    jax.lax.cond(
        jnp.logical_and(is_training, step % interval == 0),
        lambda: jax.debug.callback(callback, step, seed, metrics),
        lambda: None,
    )
