# Tab. 6: entropy of the visited-state distribution, SAC and ReMAC (M = 1, 2, 4, B = 16), 5 seeds.
#   bash scripts/state_entropy.sh hopper
source "$(dirname "$0")/common.sh"
env=$1
export SEED=2 NUM_SEEDS=5
dumps=(--save-eval-states "eval_states/$env")
run "$env" state_sac --algorithm sac "${dumps[@]}" --name "${env}_sac"
for m in 1 2 4; do
  run "$env" "state_remac_m$m" --algorithm remax_ac --set remax_m=$m --set remax_num_samples=16 \
    "${dumps[@]}" --name "${env}_remac_m$m"
done
python state_entropy.py "eval_states/$env"
