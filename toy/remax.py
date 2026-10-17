"""The unbiased ReMax estimator: C(B, m)^{-1} sum_i C(i - 1, m - 1) q_(i), q sorted ascending."""

import jax.numpy as jnp
from jax.scipy.special import gammaln


def comb_float(n, k):
    """Binomial coefficient C(n, k), elementwise; 0 where n < k."""
    n_arr = jnp.asarray(n, dtype=jnp.float64)
    k_arr = jnp.asarray(k, dtype=jnp.float64)
    valid = n_arr >= k_arr
    safe_n = jnp.where(valid, n_arr, 0.0)
    safe_k = jnp.where(valid, k_arr, 0.0)
    log_comb = gammaln(safe_n + 1.0) - gammaln(safe_k + 1.0) - gammaln(safe_n - safe_k + 1.0)
    return jnp.where(valid, jnp.exp(log_comb), 0.0)


def compute_batch_remax(returns: jnp.ndarray, m: int) -> jnp.ndarray:
    """Estimate of E[max of m draws] from the B draws in `returns`."""
    b = returns.shape[0]
    if b < m:
        raise ValueError(f"Batch size {b} must be greater than or equal to {m}.")
    weights = comb_float(jnp.arange(b), m - 1) / comb_float(b, m)
    return jnp.sum(weights * jnp.sort(returns))
