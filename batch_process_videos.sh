#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# 1) Ensure conda functions are available
#    Adjust this path if your conda is installed elsewhere.
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

# 3) Define your input/output dirs
INPUT_DIR="/home/alex/Documents/dt_ag/dt_ag/experiments/final_videos_all"
OUTPUT_BASE="/home/alex/dt_colmap_min_yolov12_appended_942_895"
mkdir -p "$OUTPUT_BASE"

# 4) Loop through videos
for filepath in "$INPUT_DIR"/*.mkv; do
  [[ -e "$filepath" ]] || continue

  filename=$(basename "$filepath")
  name="${filename%.*}"
  output_dir="$OUTPUT_BASE/$name"

  echo "→ Processing '$filename' → '$output_dir'"
  mkdir -p "$output_dir"

  ns-process-data video-bruisefacto \
    --data "$filepath" \
    --output-dir "$output_dir" \
    --skip-colmap \
    --skip-image-processing \
    --skip-gsam
done
