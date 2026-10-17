# Figs. 3, 4 and 19: SAC, PPO, and ReMAC with M = 1, 2, 4, 8.   bash scripts/main.sh hopper
source "$(dirname "$0")/common.sh"
env=$1
run "$env" sac --algorithm sac
run "$env" ppo --algorithm ppo
for m in 1 2 4 8; do
  run "$env" "remac_m$m" --algorithm remax_ac --set remax_m=$m --set remax_num_samples="$(b_for_m $m)"
done
