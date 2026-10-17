# Tab. 4, Tab. 5, Fig. 18: ReMAC (M = 1, 2, 4) with a plain-SGD actor; the Adam runs are main.sh.
#   bash scripts/sgd_actor.sh hopper
source "$(dirname "$0")/common.sh"
env=$1
for m in 1 2 4; do run "$env" "remac_sgd_m$m" --algorithm remax_ac --set remax_m=$m --set actor_optimizer=sgd; done
