#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

#── CONFIG ────────────────────────────────────────────────────────────────────────
DATA_DIR="/home/alex/dt_splats_bruise_lr_1e-3_weight_2"
EVAL_DIR="/home/alex/dt_splats_eval_bruise_lr_1e-3_weight_2"
METRICS_FILE="$EVAL_DIR/metrics.txt"
#───────────────────────────────────────────────────────────────────────────────────

mkdir -p "$EVAL_DIR"
: > "$METRICS_FILE"    # truncate or create metrics.txt

for pre_path in "$DATA_DIR"/*/; do
  pre_name=$(basename "${pre_path%/}")

  # skip any that are already post/after
  if [[ "$pre_name" == *"_after"* ]]; then
    continue
  fi

  # find matching post
  if [ -d "${DATA_DIR}/${pre_name}_after" ]; then
    post_path="${DATA_DIR}/${pre_name}_after"
  elif [ -d "${DATA_DIR}/${pre_name}_post" ]; then
    post_path="${DATA_DIR}/${pre_name}_post"
  else
    echo "[$pre_name] → WARNING: no _after/_post folder, skipping." \
      >> "$METRICS_FILE"
    continue
  fi

  # locate latest config.yml under bruisefacto/
  config_pre=$(find "$pre_path/bruisefacto" -mindepth 1 -maxdepth 1 -type d \
                  | sort | tail -n1)/config.yml
  config_post=$(find "$post_path/bruisefacto" -mindepth 1 -maxdepth 1 -type d \
                   | sort | tail -n1)/config.yml

  if [[ ! -f "$config_pre" || ! -f "$config_post" ]]; then
    echo "[$pre_name] → ERROR: missing config.yml, skipping." \
      >> "$METRICS_FILE"
    continue
  fi

  # run the eval
  out_dir="$EVAL_DIR/$pre_name"
  ns-eval-bruisefacto \
    --config-pre  "$config_pre" \
    --config-post "$config_post" \
    --output_dir  "$out_dir"

  jsonfile="$out_dir/bruise_metrics.json"
  if [[ ! -f "$jsonfile" ]]; then
    echo "[$pre_name] → ERROR: no JSON output, skipping parse." \
      >> "$METRICS_FILE"
    continue
  fi

  # extract just the three filtered-export metrics
  IFS=' ' read pre_pct post_pct diff_pct < <(
    python3 - <<PYCODE
import json, sys
data = json.load(open("$jsonfile"))
f = data["Filtered_Export_Data"]
# print all three values on one line separated by spaces, formatted to 2 decimal places
print(f"{f['Pre-Splat Bruised Points Percentage']:.2f} {f['Post-Splat Bruised Points Percentage']:.2f} {f['Bruised Points Percentage Difference']:.2f}")
PYCODE
  )

  # append labeled block to metrics.txt
  {
    echo "=== Experiment: $pre_name ==="
    printf "Pre-Splat Bruised Points Percentage: %.2f%%\n" "$pre_pct"
    printf "Post-Splat Bruised Points Percentage: %.2f%%\n" "$post_pct"
    printf "Bruised Points Percentage Difference: %.2f%%\n" "$diff_pct"
    echo
  } >> "$METRICS_FILE"

done
