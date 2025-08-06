#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# 1) Ensure conda functions are available
CONDA_BASE="${HOME}/miniconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "ERROR: cannot find conda.sh at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi

# 2) Activate your nerfsam env
conda activate nerfsam

# 3) Where your processed-video folders live
DATA_BASE="/home/alex/dt_colmap_min"
# 4) Where you want all training outputs to go
OUTPUT_DIR="/home/alex/dt_splats_min"
mkdir -p "$OUTPUT_DIR"

# 5) Loop through each subfolder in DATA_BASE
for data_dir in "$DATA_BASE"/*/; do
  [ -d "$data_dir" ] || continue

  name=$(basename "${data_dir%/}")     # e.g. dt_no_touch_after-1
  base="${name%-*}"                     # strip trailing "-<idx>": dt_no_touch_after

  # echo "→ Training '$base' on data '$data_dir'"
  # ns-train bruisefacto \
  #   --data "$data_dir" \
  #   --output-dir "$OUTPUT_DIR" \
  #   --experiment-name "$base" \
  #   --viewer.quit-on-train-completion True

  # Find the most recent training output directory
  latest_train_dir=$(find "$OUTPUT_DIR/$base/bruisefacto" -maxdepth 1 -type d | sort | tail -n 1)
  config_path="$latest_train_dir/config.yml"
  renders_path="$latest_train_dir/renders"
  
  echo "→ Rendering views for '$base' using config at '$config_path'"
  ns-render dataset \
    --load-config "$config_path" \
    --output-path "$renders_path" \
    --rendered-output-names rgb bruise strawberry strawberry_with_bruise rgb_with_bruise depth
done
