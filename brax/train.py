"""Train SAC, PPO or ReMAC on a Brax task, with every seed vmapped in one process.

    python train.py --config configs/hopper.yaml --algorithm remax_ac --seed 7 --num-seeds 10 \
        --set remax_m=4 --set remax_num_samples=8

Each evaluation prints "Seed i Step t, Mean episode length: l, Mean return: r"; SAC and ReMAC
also print their training metrics (entropy, gradient and update norms, the two entropy terms) at
the same steps.  With --wandb, metrics are logged as seed_<i>/<name> and mean/<name>.
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from rejax import PPO

import sparse_reacher  # noqa: F401  (registers brax/sparse_reacher)
from remax_ac import ReMaxAC
from sac import SAC
from state_entropy import collect_eval_states

ALGORITHMS = {"sac": SAC, "remax_ac": ReMaxAC, "ppo": PPO}


def parse_value(text):
    value = yaml.safe_load(text)
    if isinstance(value, str):  # YAML reads e.g. 1e-4 as a string
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
    return value


def load_config(path, algorithm, overrides):
    with open(path) as f:
        config = dict(yaml.safe_load(f)[algorithm])
    for item in overrides:
        key, value = item.split("=", 1)
        config[key.strip()] = parse_value(value.strip())
    return config


class Logger:
    """Prints evaluations and training metrics; logs both to wandb if a run is given."""

    def __init__(self, algorithm, num_seeds, run=None):
        self.algorithm, self.num_seeds, self.run = algorithm, num_seeds, run
        self.train = {}  # seed -> latest training metrics, logged with the next evaluation
        self.pending = {}  # step -> {name: {seed: value}}, for the mean over seeds

    def train_callback(self, step, seed, metrics):
        step, seed = int(step), int(seed)
        metrics = {f"train/{k}": float(v) for k, v in metrics.items()}
        print(f"[{self.algorithm}] seed={seed} step={step} "
              + " ".join(f"{k[6:]}={v:.6g}" for k, v in metrics.items()))
        self.train[seed] = metrics

    def eval_callback(self, step, seed, length, ret):
        step, seed = int(step), int(seed)
        values = {"eval/episode_length": float(length), "eval/return": float(ret), **self.train.pop(seed, {})}
        if self.run is None:
            return
        self.run.log({f"seed_{seed}/{k}": v for k, v in values.items()}, step=step)
        bucket = self.pending.setdefault(step, {})
        for k, v in values.items():
            bucket.setdefault(k, {})[seed] = v
        if all(len(v) == self.num_seeds for v in bucket.values()):
            self.run.log({f"mean/{k}": float(np.mean(list(v.values()))) for k, v in bucket.items()}, step=step)
            del self.pending[step]


def eval_interval(algo):
    """The step interval at which evaluations actually happen (eval_freq rounded up to whole iterations)."""
    per_iteration = algo.num_envs * getattr(algo, "num_steps", 1)
    return -(-algo.eval_freq // per_iteration) * per_iteration


def save_eval_states(eval_callback, algo, out_dir, name, keep=500, num_envs=64):
    """Wraps eval_callback to save `keep` raw observations visited by the evaluation policy."""
    os.makedirs(out_dir, exist_ok=True)
    max_steps = int(algo.env_params.max_steps_in_episode)

    def dump(step, seed, obs, n_valid):
        if int(n_valid) >= keep:
            np.savez_compressed(os.path.join(out_dir, f"{name}_seed{int(seed)}_step{int(step)}.npz"),
                                obs=np.asarray(obs, dtype=np.float32))

    def wrapped(algo, ts, rng):
        out = eval_callback(algo, ts, rng)
        rng_roll, rng_sub = jax.random.split(jax.random.fold_in(rng, 12345))
        obs, valid = collect_eval_states(algo.make_act(ts), rng_roll, algo.env, algo.env_params,
                                         num_envs, max_steps)
        obs, valid = obs.reshape(-1, obs.shape[-1]), valid.reshape(-1)
        # a uniform subsample of the valid states, without replacement
        scores = jnp.where(valid, jax.random.gumbel(rng_sub, valid.shape), -jnp.inf)
        sub = obs[jnp.argsort(-scores)[:keep]]
        jax.debug.callback(dump, ts.global_step, jax.lax.axis_index("seed"), sub, jnp.sum(valid))
        return out

    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config entry")
    ap.add_argument("--name", default=None, help="run name (wandb, and prefix of the saved eval states)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="remac")
    ap.add_argument("--save-eval-states", default=None, metavar="DIR")
    args = ap.parse_args()

    config = load_config(args.config, args.algorithm, args.set)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.name,
                         config={**config, "algorithm": args.algorithm, "seed": args.seed,
                                 "num_seeds": args.num_seeds})

    cls = ALGORITHMS[args.algorithm]
    algo = cls.create(**config)
    logger = Logger(args.algorithm, args.num_seeds, run)
    if hasattr(algo, "log_callback"):
        algo = algo.replace(log_interval=eval_interval(algo), log_callback=logger.train_callback)

    base_eval = algo.eval_callback

    def eval_callback(algo, ts, rng):
        lengths, returns = base_eval(algo, ts, rng)
        seed = jax.lax.axis_index("seed")
        jax.debug.print("Seed {} Step {}, Mean episode length: {}, Mean return: {}",
                        seed, ts.global_step, lengths.mean(), returns.mean())
        jax.debug.callback(logger.eval_callback, ts.global_step, seed, lengths.mean(), returns.mean())
        return lengths, returns

    if args.save_eval_states:
        eval_callback = save_eval_states(eval_callback, algo, args.save_eval_states,
                                         args.name or f"{args.algorithm}_{args.seed}")
    algo = algo.replace(eval_callback=eval_callback)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_seeds)
    train = jax.jit(jax.vmap(cls.train, in_axes=(None, 0), axis_name="seed"))
    _, (_, returns) = train(algo, keys)
    print(f"Final mean return per seed: {np.asarray(returns.mean(axis=-1)[:, -1])}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
