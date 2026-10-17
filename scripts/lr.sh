# Figs. 15, 16: ReMAC (M = 4) with the learning rate raised at epsilon = 1 and lowered at
# epsilon = 1e-8.   bash scripts/lr.sh halfcheetah
source "$(dirname "$0")/common.sh"
env=$1
for lr in 1e-4 3e-4 5e-4 1e-3; do
  run "$env" "remac_m4_eps1_lr$lr" --algorithm remax_ac --set actor_epsilon=1 --set learning_rate=$lr
done
for lr in 1e-5 3e-5 5e-5 1e-4; do
  run "$env" "remac_m4_lr$lr" --algorithm remax_ac --set learning_rate=$lr
done
