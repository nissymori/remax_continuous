# Fig. 17: ReMAC at the other action-sample batch size (B = 16 for M <= 4, B = 8 for M = 8).
#   bash scripts/bs.sh hopper
source "$(dirname "$0")/common.sh"
env=$1
for m in 1 2 4; do run "$env" "remac_m${m}_b16" --algorithm remax_ac --set remax_m=$m --set remax_num_samples=16; done
run "$env" remac_m8_b8 --algorithm remax_ac --set remax_m=8 --set remax_num_samples=8
