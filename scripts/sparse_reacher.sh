# Tab. 7: SAC and ReMAC (M = 1, 2, 4, 8) on the sparse-reward Reacher.   bash scripts/sparse_reacher.sh
source "$(dirname "$0")/common.sh"
run sparse_reacher sac --algorithm sac
for m in 1 2 4 8; do
  run sparse_reacher "remac_m$m" --algorithm remax_ac --set remax_m=$m --set remax_num_samples="$(b_for_m $m)"
done
