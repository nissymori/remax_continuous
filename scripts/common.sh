# Sourced by the run scripts: work in brax/ and define run and b_for_m.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../brax"

# the action-sample batch size of the paper: B = 8 for M <= 4 and B = 16 for M = 8
b_for_m () { if [ "$1" = 8 ]; then echo 16; else echo 8; fi; }

# run <env> <name> <train.py args...>: SEED (default 7) with NUM_SEEDS (default 10) seeds in one
# process, stdout to logs/<env>/<name>.log.  HumanoidStandup runs its 10 seeds (2..11) as one
# process each, to fit in GPU memory.
run () {
  local env=$1 name=$2; shift 2
  mkdir -p "logs/$env"
  if [ "$env" = humanoidstandup ]; then
    for s in $(seq 2 11); do
      python train.py --config "configs/$env.yaml" --seed "$s" --num-seeds 1 "$@" > "logs/$env/${name}_seed$s.log"
    done
  else
    python train.py --config "configs/$env.yaml" --seed "${SEED:-7}" --num-seeds "${NUM_SEEDS:-10}" "$@" \
      > "logs/$env/$name.log"
  fi
}
