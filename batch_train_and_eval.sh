#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ---------- STATIC PATHS ----------
COLMAP_BASE="/home/alex/dt_colmap_min_yolov12_appended_942_895"
OUTPUT_ROOT="/home/alex/dt_yolov12_942_895_final"

# ---------- CONDA ---------------
CONDA_BASE="${HOME}/miniconda3"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "ERROR: cannot find conda.sh at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate nerfsam

# ---------- GRID SEARCH ----------
for BT in $(seq 0.2 0.05 0.90); do
  printf -v BT_STR "%.2f" "$BT" # two-decimal format

  echo "═════════════════════════════════════════════════════"
  echo "▶ bruise-threshold: $BT_STR"
  echo "═════════════════════════════════════════════════════"

  # ---------- FOLDER SET-UP ----------
  PARENT_DIR="${OUTPUT_ROOT}/dt_train_eval_bt_${BT_STR}"
  SPLAT_BASE="${PARENT_DIR}/splats"
  EVAL_BASE="${PARENT_DIR}/evals"
  mkdir -p "$SPLAT_BASE" "$EVAL_BASE"

  # ---------- ① TRAIN PHASE ----------
  echo "=== [TRAIN] datasets in '$COLMAP_BASE' ==="
  for data_dir in "$COLMAP_BASE"/*/; do
    [[ -d "$data_dir" ]] || continue
    scene_name=$(basename "${data_dir%/}")     # eg. dt_no_touch_after-1

    echo " → Training scene '$scene_name'"
    ns-train bruisefacto \
      --data "$data_dir" \
      --output-dir "$SPLAT_BASE" \
      --viewer.quit-on-train-completion True
  done

  # ---------- ② RENDER PHASE ----------
  echo "=== [RENDER] generating views for all trained splats ==="
  for splat_dir in "$SPLAT_BASE"/*/; do
    [[ -d "$splat_dir" ]] || continue
    splat_name=$(basename "${splat_dir%/}")

    # Find the most recent training output directory
    latest_train_dir=$(find "$splat_dir/bruisefacto" -maxdepth 1 -type d | sort | tail -n 1)
    config_path="$latest_train_dir/config.yml"
    renders_path="$latest_train_dir/renders"
    
    [[ -f "$config_path" ]] || { echo "WARNING: no config.yml for '$splat_name', skipping render."; continue; }
    
    echo " → Rendering views for '$splat_name'"
    ns-render dataset \
      --load-config "$config_path" \
      --output-path "$renders_path" \
      --rendered-output-names rgb bruise strawberry strawberry_with_bruise rgb_with_bruise depth
  done

  # ---------- ③ EVAL PHASE ----------
  echo "=== [EVAL] threshold = $BT_STR ==="
  METRICS_FILE="$EVAL_BASE/metrics_bt_${BT_STR}.txt"
  : > "$METRICS_FILE"

  for pre_path in "$SPLAT_BASE"/*/; do
    pre_name=$(basename "${pre_path%/}")

    [[ "$pre_name" == *_after ]] && continue

    if   [[ -d "${SPLAT_BASE}/${pre_name}_after" ]]; then
      post_path="${SPLAT_BASE}/${pre_name}_after"
    elif [[ -d "${SPLAT_BASE}/${pre_name}_post"  ]]; then
      post_path="${SPLAT_BASE}/${pre_name}_post"
    else
      echo "[$pre_name] → WARNING: missing _after/_post, skipping." >>"$METRICS_FILE"
      continue
    fi

    config_pre=$(find "$pre_path/bruisefacto"  -mindepth 1 -maxdepth 1 -type d | sort | tail -n1)/config.yml
    config_post=$(find "$post_path/bruisefacto" -mindepth 1 -maxdepth 1 -type d | sort | tail -n1)/config.yml
    [[ -f "$config_pre" && -f "$config_post" ]] || { 
      echo "[$pre_name] → ERROR: config.yml missing, skipping." >>"$METRICS_FILE"; continue; }

    out_dir="$EVAL_BASE/$pre_name"
    mkdir -p "$out_dir"
    echo " → Evaluating '$pre_name'"

    ns-eval-bruisefacto \
      --config-pre  "$config_pre" \
      --config-post "$config_post" \
      --output_dir  "$out_dir" \
      --bruise-threshold "$BT_STR"

    jsonfile="$out_dir/bruise_metrics.json"
    [[ -f "$jsonfile" ]] || { 
      echo "[$pre_name] → ERROR: no JSON output." >>"$METRICS_FILE"; continue; }

    # Pull three key numbers from JSON
    IFS=' ' read pre_pct post_pct diff_pct < <(
      python3 - <<PY
import json, sys
f=json.load(open("$jsonfile"))["Filtered_Export_Data"]
print(f"{f['Pre-Splat Bruised Points Percentage']:.2f}",
      f"{f['Post-Splat Bruised Points Percentage']:.2f}",
      f"{f['Bruised Points Percentage Difference']:.2f}")
PY
      )

      {
        echo "=== $pre_name (BT=$BT_STR) ==="
        printf "Pre-Splat  : %s%%\n" "$pre_pct"
        printf "Post-Splat : %s%%\n" "$post_pct"
        printf "Difference : %s%%\n\n" "$diff_pct"
      } >>"$METRICS_FILE"
    done

    echo "✔ Finished combo BT=$BT_STR"
    echo "  → Metrics at: $METRICS_FILE"
  done
done

echo "=== GRID SEARCH COMPLETE ==="
