# Retry Policy Gradients in Continuous Action Spaces

Code for *Retry Policy Gradients in Continuous Action Spaces* (Soichiro Nishimori and Paavo
Parmas, TMLR 2026, [OpenReview](https://openreview.net/forum?id=qmCljotj08)).

**ReMAC** (ReMax Actor-Critic) is SAC's off-policy actor-critic without entropy regularization,
whose actor maximizes the ReMax objective E[max of M Q-values of the policy's own actions],
estimated from B >= M reparameterized action samples per state.

```
brax/     ReMAC, SAC and the training script (PPO is rejax's), configs, sparse Reacher,
          visited-state entropy
scripts/  one script per experiment of the paper
toy/      the toy problem of Sec. 5
```

## Setup

```bash
pip install -r requirements.txt   # install JAX for your accelerator, e.g. "jax[cuda12]==0.6.0"
```

## Brax experiments

A single run, with 10 seeds vmapped in one process:

```bash
cd brax
python train.py --config configs/hopper.yaml --algorithm remax_ac --seed 7 --num-seeds 10 \
    --set remax_m=4 --set remax_num_samples=8
```

The configs hold the paper's settings: rejax's tuned SAC and PPO hyperparameters, ReMAC's tuned
learning rate (Tab. 2), 3M environment steps and an evaluation every 30k steps.  `--wandb` logs
to Weights & Biases.  Each script below runs the configurations of one experiment, one after
another, and writes the logs to `brax/logs/<task>/`.  Tasks: `ant`, `halfcheetah`, `hopper`,
`walker2d`, `reacher`, `swimmer`, `humanoidstandup`.

| Paper | Command |
|---|---|
| Figs. 3, 4, 19 | `bash scripts/main.sh <task>` |
| Tab. 2 | `bash scripts/tune.sh <task>` |
| Figs. 7–14 | `bash scripts/eps.sh <task>` |
| Figs. 15, 16 | `bash scripts/lr.sh <task>` (`halfcheetah`, `swimmer`, `walker2d`) |
| Fig. 17 | `bash scripts/bs.sh <task>` |
| Tabs. 4, 5, Fig. 18 | `bash scripts/sgd_actor.sh <task>` (`halfcheetah`, `ant`, `hopper`) |
| Tab. 6 | `bash scripts/state_entropy.sh <task>` (`ant`, `halfcheetah`, `hopper`) |
| Tab. 7 | `bash scripts/sparse_reacher.sh` |

The final value of a run is the mean over the last 10% of training.

## Toy problem

```bash
bash scripts/toy.sh   # Figs. 1, 2, 5, 6 and Tab. 1, into toy/fig/
```

## Citation

```bibtex
@article{nishimori2026retry,
  title   = {Retry Policy Gradients in Continuous Action Spaces},
  author  = {Nishimori, Soichiro and Parmas, Paavo},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=qmCljotj08}
}
```
