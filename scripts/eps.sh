# Figs. 7-14: ReMAC with Adam's epsilon in {1e-2, 1e-1, 1} (1e-8 is main.sh).   bash scripts/eps.sh hopper
source "$(dirname "$0")/common.sh"
env=$1
for eps in 1e-2 1e-1 1; do
  for m in 1 2 4 8; do
    run "$env" "remac_m${m}_eps$eps" --algorithm remax_ac --set remax_m=$m \
      --set remax_num_samples="$(b_for_m $m)" --set actor_epsilon=$eps
  done
done
