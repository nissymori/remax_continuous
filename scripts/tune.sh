# Tab. 2: ReMAC's learning-rate sweep, 3 seeds, evaluated every 100k steps.   bash scripts/tune.sh hopper
source "$(dirname "$0")/common.sh"
env=$1
export SEED=1 NUM_SEEDS=3
for lr in 1e-4 2e-4 3e-4 5e-4 1e-3; do
  for m in 1 2 4 8; do
    run "$env" "tune_m${m}_lr$lr" --algorithm remax_ac --set remax_m=$m \
      --set remax_num_samples="$(b_for_m $m)" --set learning_rate=$lr --set eval_freq=100000
  done
done
