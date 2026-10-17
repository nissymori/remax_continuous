# The toy problem (Figs. 1, 2, 5, 6 and Tab. 1); figures are written to toy/fig/.   bash scripts/toy.sh
set -euo pipefail
cd "$(dirname "$0")/../toy"
# Fig. 1: the ReMax gradient field over (mu, sigma)
python quad_vector_field.py --m 1 --alpha 0.0 --normalize --add-y-label --plot-format pdf --no-colorbar
python quad_vector_field.py --m 1 --alpha 0.1 --normalize --plot-format pdf --no-colorbar
python quad_vector_field.py --m 4 --alpha 0.0 --normalize --plot-format pdf --no-colorbar
python quad_vector_field.py --m 8 --alpha 0.0 --normalize --plot-format pdf --colorbar
# Figs. 2 and 5: optimization paths and distance to the optimum under Adam and SGD
for m in 1 2 4 8; do
  case $m in 1) extra=(--add-y-label) ;; 2|4) extra=(--no-legend) ;; *) extra=() ;; esac
  python quad_opt_update_norm.py --m $m --distance-only "${extra[@]}"
  python quad_opt_update_norm.py --m $m --optimize-only "${extra[@]}"
done
# Fig. 6 and Tab. 1: removing Adam's denominator
python adam_denominator.py --save fig/adam_denominator.pdf
