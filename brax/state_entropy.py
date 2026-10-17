"""Entropy of the visited-state distribution (App. F.2).

train.py --save-eval-states DIR writes, at every evaluation, 500 raw observations visited by the
evaluation policy in 64 environments, as DIR/<run name>_seed<i>_step<t>.npz.  This script pools,
for every seed, the states of all evaluations, standardizes each environment's observations with
one reference mean and standard deviation shared by every method, and reports the k-nearest-
neighbor entropy estimate of Mutti et al. (2021) as mean +- standard error over seeds.

    python state_entropy.py eval_states/          # run names <env>_<method>, e.g. hopper_remac_m4
"""

import argparse
import glob
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

FNAME = re.compile(r"(?P<env>[a-z0-9]+)_(?P<method>[a-z0-9_]+)_seed(?P<seed>\d+)_step(?P<step>\d+)\.npz")
POOL_PER_SEED = 2000  # states each (method, seed) contributes to the reference statistics


def _rollout(act, env, env_params, rng, max_steps):
    rng_reset, rng = jax.random.split(rng)
    obs0, state0 = env.reset(rng_reset, env_params)

    def step(carry, _):
        rng, state, obs, done = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        next_obs, state, _, done_step, _ = env.step(rng_step, state, act(obs, rng_act), env_params)
        return (rng, state, next_obs, jnp.logical_or(done, done_step)), (obs, jnp.logical_not(done))

    _, (obs, valid) = jax.lax.scan(step, (rng, state0, obs0, jnp.array(False)), None, length=max_steps)
    return obs, valid


def collect_eval_states(act, rng, env, env_params, num_envs, max_steps):
    """Raw observations of `num_envs` evaluation rollouts, and a mask of the steps before termination."""
    return jax.vmap(_rollout, in_axes=(None, None, None, 0, None))(
        act, env, env_params, jax.random.split(rng, num_envs), max_steps
    )


def knn_entropy(x, k=4):
    """k-NN differential-entropy estimate (nats) of the distribution of the rows of x."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    dist, _ = cKDTree(x).query(x, k=k + 1, workers=-1)
    log_v = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0) + d * np.log(np.maximum(dist[:, k], 1e-12))
    return float(-(np.log(k) - np.log(n) - log_v.mean()) + np.log(k) - digamma(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states_dir")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    runs = {}
    for path in glob.glob(os.path.join(args.states_dir, "*.npz")):
        m = FNAME.match(os.path.basename(path))
        if m:
            runs.setdefault((m["env"], m["method"], int(m["seed"])), {})[int(m["step"])] = path
    for env in sorted({e for e, _, _ in runs}):
        env_runs = {key: v for key, v in sorted(runs.items()) if key[0] == env}
        steps = sorted(set.intersection(*(set(v) for v in env_runs.values())))
        pooled = {key: np.concatenate([np.load(v[s])["obs"] for s in steps]) for key, v in env_runs.items()}
        rng = np.random.default_rng(0)
        ref = np.concatenate([x[rng.choice(len(x), min(POOL_PER_SEED, len(x)), replace=False)]
                              for x in pooled.values()])
        mean, std = ref.mean(0), ref.std(0)
        keep = std > 1e-8
        print(f"{env}: {len(steps)} evaluations, {len(next(iter(pooled.values())))} states per seed")
        per_method = {}
        for (_, method, _), x in pooled.items():
            per_method.setdefault(method, []).append(knn_entropy((x - mean)[:, keep] / std[keep], args.k))
        for method, v in per_method.items():
            v = np.asarray(v)
            print(f"  {method:12s} {v.mean():7.2f} +- {v.std(ddof=1) / np.sqrt(len(v)):.2f}  (n={len(v)})")


if __name__ == "__main__":
    main()
