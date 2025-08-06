#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ---------- STATIC PATHS ----------
COLMAP_BASE="/home/alex/dt_colmap_min_yolov12_appended_942_895"
OUTPUT_ROOT="/home/alex/dt_yolov12_942_895"

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
for BW in $(seq 1 0.5 3); do
  printf -v BW_STR "%.1f" "$BW"   # "1.0" → "1.0", "1.5" → "1.5"
  for BT in $(seq 0.3 0.1 0.50); do
    printf -v BT_STR "%.2f" "$BT" # two-decimal format

    echo "═════════════════════════════════════════════════════"
    echo "▶ bruise-weight   : $BW_STR"
    echo "▶ bruise-threshold: $BT_STR"
    echo "═════════════════════════════════════════════════════"

    # ---------- FOLDER SET-UP ----------
    PARENT_DIR="${OUTPUT_ROOT}/dt_train_eval_bw_${BW_STR}_bt_${BT_STR}"
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
        --pipeline.model.bruise-weight "$BW_STR" \
        --viewer.quit-on-train-completion True
    done

    # ---------- ② EVAL PHASE ----------
    echo "=== [EVAL] threshold = $BT_STR ==="
    METRICS_FILE="$EVAL_BASE/metrics_bw_${BW_STR}_bt_${BT_STR}.txt"
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
        echo "=== $pre_name (BW=$BW_STR  BT=$BT_STR) ==="
        printf "Pre-Splat  : %s%%\n" "$pre_pct"
        printf "Post-Splat : %s%%\n" "$post_pct"
        printf "Difference : %s%%\n\n" "$diff_pct"
      } >>"$METRICS_FILE"
    done

    echo "✔ Finished combo BW=$BW_STR  BT=$BT_STR"
    echo "  → Metrics at: $METRICS_FILE"
  done
done

echo "=== GRID SEARCH COMPLETE ==="
