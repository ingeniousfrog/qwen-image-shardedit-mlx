#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_MODEL="$ROOT_DIR/models/qwen-edit-2511-q6"
DEFAULT_LORA="$ROOT_DIR/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
DEFAULT_IMAGE="$ROOT_DIR/ref.png"
DEFAULT_PROMPT="Replace the background with a naturally lit cafe interior while preserving the subject identity, hair, expression, pose, and clothing details."

RUNTIME="mflux"
MODEL_PATH="${SHARDEDIT_MODEL_PATH:-$DEFAULT_MODEL}"
LORA_PATH="${SHARDEDIT_LORA_PATH:-$DEFAULT_LORA}"
IMAGE_PATHS=("${SHARDEDIT_IMAGE_PATH:-$DEFAULT_IMAGE}")
IMAGE_OPTION_SEEN=0
PROMPT="${SHARDEDIT_PROMPT:-$DEFAULT_PROMPT}"
SEED="${SHARDEDIT_SEED:-42}"
WIDTH="${SHARDEDIT_WIDTH:-576}"
HEIGHT="${SHARDEDIT_HEIGHT:-768}"
STEPS="${SHARDEDIT_STEPS:-8}"
GUIDANCE="${SHARDEDIT_GUIDANCE:-1.0}"
TRUE_CFG_SCALE="${SHARDEDIT_TRUE_CFG_SCALE:-1.0}"
REPEATS="${SHARDEDIT_REPEATS:-1}"
OUTPUT_ROOT="${SHARDEDIT_OUTPUT_ROOT:-$ROOT_DIR/benchmark-runs}"
MFLUX_BIN="${MFLUX_BIN:-mflux-generate-qwen-edit}"
SHARDEDIT_MFLUX_BIN="${SHARDEDIT_MFLUX_BIN:-$ROOT_DIR/.venv/bin/shardedit-mflux-edit}"
MFLUX_CACHE_LIMIT_GB="${SHARDEDIT_MFLUX_CACHE_LIMIT_GB:-}"
EVAL_EVERY_N_BLOCKS="${SHARDEDIT_EVAL_EVERY_N_BLOCKS:-0}"
PROBE_BLOCKS="${SHARDEDIT_PROBE_BLOCKS:-}"
TOKEN_REDUNDANCY_BLOCKS="${SHARDEDIT_TOKEN_REDUNDANCY_BLOCKS:-}"
TOKEN_REDUNDANCY_HEATMAP_DIR="${SHARDEDIT_TOKEN_REDUNDANCY_HEATMAP_DIR:-}"
BRIDGE_ERROR_DIAGNOSE="${SHARDEDIT_BRIDGE_ERROR_DIAGNOSE:-0}"
BRIDGE_ERROR_HEATMAP_DIR="${SHARDEDIT_BRIDGE_ERROR_HEATMAP_DIR:-}"
SELECTIVE_REFILL_FRACTION="${SHARDEDIT_SELECTIVE_REFILL_FRACTION:-0}"
SELECTIVE_REFILL_MODE="${SHARDEDIT_SELECTIVE_REFILL_MODE:-residual-dampen}"
SELECTIVE_REFILL_DAMPEN="${SHARDEDIT_SELECTIVE_REFILL_DAMPEN:-1.0}"
SELECTIVE_REFILL_MIN_STEP="${SHARDEDIT_SELECTIVE_REFILL_MIN_STEP:-0}"
CACHE_THRESHOLD="${SHARDEDIT_CACHE_THRESHOLD:-0}"
CACHE_MAX_CONSECUTIVE="${SHARDEDIT_CACHE_MAX_CONSECUTIVE:-1}"
CACHE_WARMUP_STEPS="${SHARDEDIT_CACHE_WARMUP_STEPS:-1}"
CACHE_BACK_BLOCKS="${SHARDEDIT_CACHE_BACK_BLOCKS:-0}"
CACHE_ANCHOR_MODE="${SHARDEDIT_CACHE_ANCHOR_MODE:-residual}"
CACHE_PREDICTOR="${SHARDEDIT_CACHE_PREDICTOR:-last}"
CACHE_THRESHOLD_SCHEDULE="${SHARDEDIT_CACHE_THRESHOLD_SCHEDULE:-fixed}"
CACHE_REGION_POLICY="${SHARDEDIT_CACHE_REGION_POLICY:-all}"
CACHE_PRESET="${SHARDEDIT_CACHE_PRESET:-custom}"
REFERENCE_CONDITIONING_SIZE="${SHARDEDIT_REFERENCE_CONDITIONING_SIZE:-fit-box}"
REFERENCE_CONDITIONING_SHORT_SIDE="${SHARDEDIT_REFERENCE_CONDITIONING_SHORT_SIDE:-512}"
REFERENCE_CONDITIONING_MAX_WIDTH="${SHARDEDIT_REFERENCE_CONDITIONING_MAX_WIDTH:-576}"
REFERENCE_CONDITIONING_MAX_HEIGHT="${SHARDEDIT_REFERENCE_CONDITIONING_MAX_HEIGHT:-768}"
RESIDENCY_MODE="${SHARDEDIT_RESIDENCY:-shard}"
RESIDENCY_WINDOW_SIZE="${SHARDEDIT_RESIDENCY_WINDOW_SIZE:-8}"
RELEASE_POLICY="${SHARDEDIT_RELEASE_POLICY:-window}"
DENSE_IMG_FF_WINDOW="${SHARDEDIT_DENSE_IMG_FF_WINDOW:-0}"
DENSE_IMG_FF_CACHE_MAX_BLOCKS="${SHARDEDIT_DENSE_IMG_FF_CACHE_MAX_BLOCKS:-60}"
KQUANT_IMG_FF_WINDOW="${SHARDEDIT_KQUANT_IMG_FF_WINDOW:-0}"
KQUANT_IMG_FF_CACHE_MAX_BLOCKS="${SHARDEDIT_KQUANT_IMG_FF_CACHE_MAX_BLOCKS:-60}"
KQUANT_IMG_FF_CODEC="${SHARDEDIT_KQUANT_IMG_FF_CODEC:-q6_k}"
LORA_TENSOR_CACHE="${SHARDEDIT_LORA_TENSOR_CACHE:-0}"
LORA_TENSOR_CACHE_MAX_WINDOWS="${SHARDEDIT_LORA_TENSOR_CACHE_MAX_WINDOWS:-8}"
PATCHED_WINDOW_CACHE_MAX_WINDOWS="${SHARDEDIT_PATCHED_WINDOW_CACHE_MAX_WINDOWS:-0}"
CONDITION_TOKEN_MERGE="${SHARDEDIT_CONDITION_TOKEN_MERGE:-0}"
CONDITION_TOKEN_MERGE_STRIDE="${SHARDEDIT_CONDITION_TOKEN_MERGE_STRIDE:-2}"
CONDITION_TOKEN_MERGE_START_BLOCK="${SHARDEDIT_CONDITION_TOKEN_MERGE_START_BLOCK:-2}"
CONDITION_TOKEN_MERGE_BACK_BLOCKS="${SHARDEDIT_CONDITION_TOKEN_MERGE_BACK_BLOCKS:-2}"
TEXT_TOKEN_MERGE="${SHARDEDIT_TEXT_TOKEN_MERGE:-0}"
TEXT_TOKEN_MERGE_STRIDE="${SHARDEDIT_TEXT_TOKEN_MERGE_STRIDE:-2}"
TEXT_TOKEN_MERGE_START_BLOCK="${SHARDEDIT_TEXT_TOKEN_MERGE_START_BLOCK:-2}"
TEXT_TOKEN_MERGE_BACK_BLOCKS="${SHARDEDIT_TEXT_TOKEN_MERGE_BACK_BLOCKS:-2}"
Q6_LINEAR_PROFILE="${SHARDEDIT_Q6_LINEAR_PROFILE:-0}"
SHARDEDIT_PROFILE="${SHARDEDIT_PROFILE:-1}"
QWEN_IMAGE_CLI="${QWEN_IMAGE_CLI:-QwenImageCLI}"
GPU_CACHE_LIMIT="${SHARDEDIT_GPU_CACHE_LIMIT:-16gb}"
COOLDOWN_SECONDS="${SHARDEDIT_COOLDOWN_SECONDS:-0}"
RUN_SEQUENCE_LABEL="${SHARDEDIT_RUN_SEQUENCE_LABEL:-}"
CONDITION_NOTE="${SHARDEDIT_CONDITION_NOTE:-}"
THERMAL_NOTE="${SHARDEDIT_THERMAL_NOTE:-}"
LOW_RAM=1
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  benchmarks/run_qwen_edit_benchmark.sh [options]

Options:
  --runtime mflux|mflux-no-lowram|shardedit|swift|both
  --model PATH
  --lora PATH
  --image PATH (repeatable)
  --prompt TEXT
  --seed N
  --width N
  --height N
  --steps N
  --guidance FLOAT
  --true-cfg-scale FLOAT
  --repeats N
  --output-root DIR
  --mflux-bin PATH
  --shardedit-mflux-bin PATH
  --mflux-cache-limit-gb N
  --eval-every-n-blocks N
  --probe-blocks LIST
  --token-redundancy-blocks LIST
  --token-redundancy-heatmap-dir DIR
  --bridge-error-diagnose
  --bridge-error-heatmap-dir DIR
  --selective-refill-fraction FLOAT
  --selective-refill-mode {subset,subset-f1,residual-dampen,uniqueness-scale,uniqueness-boost}
  --selective-refill-dampen FLOAT
  --selective-refill-min-step INT
  --cache-threshold FLOAT
  --cache-max-consecutive N
  --cache-warmup-steps N
  --cache-back-blocks N
  --cache-anchor-mode residual|absolute
  --cache-predictor last|linear|linear-residual|quadratic|quadratic-residual|adams-bashforth|adams-bashforth-residual
  --cache-threshold-schedule fixed|sigma|flow-aware|flow-aware-veto
  --cache-region-policy all|target-conservative|condition-conservative
  --cache-preset custom|none|f1b2|flow-aware|flow-aware-veto|f1b2-linear|f1b2-ab2
  --reference-conditioning-size upstream|original|short-side|short-side-512|fit-box
  --reference-conditioning-short-side N
  --reference-conditioning-max-width N
  --reference-conditioning-max-height N
  --residency none|shard|window (default: shard for shardedit)
  --residency-window-size N
  --release-policy window|step|none|keep-last
  --dense-img-ff-window
  --dense-img-ff-cache-max-blocks N (default: 60 when dense img_ff is enabled)
  --kquant-img-ff-window (diagnostic no-go probe)
  --kquant-img-ff-cache-max-blocks N (default: 60 when K-quant img_ff is enabled)
  --kquant-img-ff-codec CODEC (default: q6_k)
  --lora-tensor-cache
  --lora-tensor-cache-max-windows N
  --patched-window-cache-max-windows N (0 disables)
  --condition-token-merge
  --condition-token-merge-stride N
  --condition-token-merge-start-block N
  --condition-token-merge-back-blocks N
  --text-token-merge
  --text-token-merge-stride N
  --text-token-merge-start-block N
  --text-token-merge-back-blocks N
  --q6-linear-profile
  --no-shardedit-profile
  --qwen-image-cli PATH
  --gpu-cache-limit VALUE
  --cooldown-seconds N
  --run-sequence-label TEXT
  --condition-note TEXT
  --thermal-note TEXT
  --dry-run
  -h, --help

Token merge flags are diagnostic-only V0 probes. The 2026-07-22 fit-box smoke
rejected them as candidate acceleration paths.

Environment overrides:
  SHARDEDIT_MODEL_PATH, SHARDEDIT_LORA_PATH, SHARDEDIT_IMAGE_PATH, SHARDEDIT_PROMPT, SHARDEDIT_SEED
  SHARDEDIT_WIDTH, SHARDEDIT_HEIGHT, SHARDEDIT_STEPS, SHARDEDIT_GUIDANCE
  SHARDEDIT_TRUE_CFG_SCALE, SHARDEDIT_REPEATS, SHARDEDIT_OUTPUT_ROOT
  MFLUX_BIN, SHARDEDIT_MFLUX_BIN, SHARDEDIT_MFLUX_CACHE_LIMIT_GB
  SHARDEDIT_EVAL_EVERY_N_BLOCKS, SHARDEDIT_PROBE_BLOCKS, SHARDEDIT_TOKEN_REDUNDANCY_BLOCKS
  SHARDEDIT_TOKEN_REDUNDANCY_HEATMAP_DIR
  SHARDEDIT_BRIDGE_ERROR_DIAGNOSE, SHARDEDIT_BRIDGE_ERROR_HEATMAP_DIR
  SHARDEDIT_SELECTIVE_REFILL_FRACTION, SHARDEDIT_SELECTIVE_REFILL_MODE
  SHARDEDIT_SELECTIVE_REFILL_DAMPEN, SHARDEDIT_SELECTIVE_REFILL_MIN_STEP
  SHARDEDIT_CACHE_THRESHOLD
  SHARDEDIT_CACHE_MAX_CONSECUTIVE, SHARDEDIT_CACHE_WARMUP_STEPS, SHARDEDIT_CACHE_BACK_BLOCKS
  SHARDEDIT_CACHE_ANCHOR_MODE, SHARDEDIT_CACHE_PREDICTOR, SHARDEDIT_CACHE_THRESHOLD_SCHEDULE
  SHARDEDIT_CACHE_REGION_POLICY, SHARDEDIT_CACHE_PRESET
  SHARDEDIT_REFERENCE_CONDITIONING_SIZE, SHARDEDIT_REFERENCE_CONDITIONING_SHORT_SIDE
  SHARDEDIT_REFERENCE_CONDITIONING_MAX_WIDTH, SHARDEDIT_REFERENCE_CONDITIONING_MAX_HEIGHT
  SHARDEDIT_RESIDENCY, SHARDEDIT_RESIDENCY_WINDOW_SIZE, SHARDEDIT_RELEASE_POLICY
  SHARDEDIT_DENSE_IMG_FF_WINDOW, SHARDEDIT_DENSE_IMG_FF_CACHE_MAX_BLOCKS
  SHARDEDIT_KQUANT_IMG_FF_WINDOW, SHARDEDIT_KQUANT_IMG_FF_CACHE_MAX_BLOCKS
  SHARDEDIT_KQUANT_IMG_FF_CODEC
  SHARDEDIT_LORA_TENSOR_CACHE, SHARDEDIT_LORA_TENSOR_CACHE_MAX_WINDOWS
  SHARDEDIT_PATCHED_WINDOW_CACHE_MAX_WINDOWS
  SHARDEDIT_CONDITION_TOKEN_MERGE, SHARDEDIT_CONDITION_TOKEN_MERGE_STRIDE
  SHARDEDIT_CONDITION_TOKEN_MERGE_START_BLOCK, SHARDEDIT_CONDITION_TOKEN_MERGE_BACK_BLOCKS
  SHARDEDIT_TEXT_TOKEN_MERGE, SHARDEDIT_TEXT_TOKEN_MERGE_STRIDE
  SHARDEDIT_TEXT_TOKEN_MERGE_START_BLOCK, SHARDEDIT_TEXT_TOKEN_MERGE_BACK_BLOCKS
  SHARDEDIT_Q6_LINEAR_PROFILE
  SHARDEDIT_PROFILE
  QWEN_IMAGE_CLI, SHARDEDIT_GPU_CACHE_LIMIT
  SHARDEDIT_COOLDOWN_SECONDS, SHARDEDIT_RUN_SEQUENCE_LABEL
  SHARDEDIT_CONDITION_NOTE, SHARDEDIT_THERMAL_NOTE
USAGE
}

fail() {
  echo "error: $*" >&2
  exit 2
}

apply_cache_preset() {
  case "$CACHE_PRESET" in
    custom) ;;
    none)
      CACHE_THRESHOLD="0"
      CACHE_MAX_CONSECUTIVE="1"
      CACHE_WARMUP_STEPS="1"
      CACHE_BACK_BLOCKS="0"
      CACHE_ANCHOR_MODE="residual"
      CACHE_PREDICTOR="last"
      CACHE_THRESHOLD_SCHEDULE="fixed"
      CACHE_REGION_POLICY="all"
      ;;
    f1b2)
      CACHE_THRESHOLD="0.8"
      CACHE_MAX_CONSECUTIVE="1"
      CACHE_WARMUP_STEPS="1"
      CACHE_BACK_BLOCKS="2"
      CACHE_ANCHOR_MODE="residual"
      CACHE_PREDICTOR="last"
      CACHE_THRESHOLD_SCHEDULE="fixed"
      CACHE_REGION_POLICY="all"
      ;;
    flow-aware)
      CACHE_THRESHOLD="0.8"
      CACHE_MAX_CONSECUTIVE="1"
      CACHE_WARMUP_STEPS="1"
      CACHE_BACK_BLOCKS="2"
      CACHE_ANCHOR_MODE="residual"
      CACHE_PREDICTOR="last"
      CACHE_THRESHOLD_SCHEDULE="flow-aware"
      CACHE_REGION_POLICY="all"
      ;;
    flow-aware-veto)
      CACHE_THRESHOLD="0.8"
      CACHE_MAX_CONSECUTIVE="1"
      CACHE_WARMUP_STEPS="1"
      CACHE_BACK_BLOCKS="2"
      CACHE_ANCHOR_MODE="residual"
      CACHE_PREDICTOR="last"
      CACHE_THRESHOLD_SCHEDULE="flow-aware-veto"
      CACHE_REGION_POLICY="all"
      ;;
    f1b2-linear)
      CACHE_THRESHOLD="0.8"
      CACHE_MAX_CONSECUTIVE="1"
      CACHE_WARMUP_STEPS="1"
      CACHE_BACK_BLOCKS="2"
      CACHE_ANCHOR_MODE="residual"
      CACHE_PREDICTOR="linear"
      CACHE_THRESHOLD_SCHEDULE="fixed"
      CACHE_REGION_POLICY="all"
      ;;
    f1b2-ab2)
      CACHE_THRESHOLD="0.8"
      CACHE_MAX_CONSECUTIVE="1"
      CACHE_WARMUP_STEPS="1"
      CACHE_BACK_BLOCKS="2"
      CACHE_ANCHOR_MODE="residual"
      CACHE_PREDICTOR="adams-bashforth"
      CACHE_THRESHOLD_SCHEDULE="fixed"
      CACHE_REGION_POLICY="all"
      ;;
    *) fail "--cache-preset must be custom, none, f1b2, flow-aware, flow-aware-veto, f1b2-linear, or f1b2-ab2" ;;
  esac
}

apply_cache_env_overrides() {
  [[ -n "${SHARDEDIT_CACHE_THRESHOLD+x}" ]] && CACHE_THRESHOLD="$SHARDEDIT_CACHE_THRESHOLD"
  [[ -n "${SHARDEDIT_CACHE_MAX_CONSECUTIVE+x}" ]] && CACHE_MAX_CONSECUTIVE="$SHARDEDIT_CACHE_MAX_CONSECUTIVE"
  [[ -n "${SHARDEDIT_CACHE_WARMUP_STEPS+x}" ]] && CACHE_WARMUP_STEPS="$SHARDEDIT_CACHE_WARMUP_STEPS"
  [[ -n "${SHARDEDIT_CACHE_BACK_BLOCKS+x}" ]] && CACHE_BACK_BLOCKS="$SHARDEDIT_CACHE_BACK_BLOCKS"
  [[ -n "${SHARDEDIT_CACHE_ANCHOR_MODE+x}" ]] && CACHE_ANCHOR_MODE="$SHARDEDIT_CACHE_ANCHOR_MODE"
  [[ -n "${SHARDEDIT_CACHE_PREDICTOR+x}" ]] && CACHE_PREDICTOR="$SHARDEDIT_CACHE_PREDICTOR"
  [[ -n "${SHARDEDIT_CACHE_THRESHOLD_SCHEDULE+x}" ]] && CACHE_THRESHOLD_SCHEDULE="$SHARDEDIT_CACHE_THRESHOLD_SCHEDULE"
  [[ -n "${SHARDEDIT_CACHE_REGION_POLICY+x}" ]] && CACHE_REGION_POLICY="$SHARDEDIT_CACHE_REGION_POLICY"
}

quote_command() {
  if command -v python3 >/dev/null; then
    python3 - "$@" <<'PY'
import shlex
import sys

print(" ".join(shlex.quote(arg) for arg in sys.argv[1:]))
PY
  else
    printf '%q ' "$@"
    printf '\n'
  fi
}

need_file() {
  local label="$1"
  local path="$2"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    [[ -e "$path" ]] || echo "dry-run warning: $label does not exist yet: $path" >&2
    return 0
  fi
  [[ -e "$path" ]] || fail "$label does not exist: $path"
}

need_executable_command() {
  local label="$1"
  local cmd="$2"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ "$cmd" == */* && ! -x "$cmd" ]]; then
      echo "dry-run warning: $label is not executable yet: $cmd" >&2
    elif [[ "$cmd" != */* ]] && ! command -v "$cmd" >/dev/null; then
      echo "dry-run warning: $label not found on PATH yet: $cmd" >&2
    fi
    return 0
  fi
  if [[ "$cmd" == */* ]]; then
    [[ -x "$cmd" ]] || fail "$label is not executable: $cmd"
  else
    command -v "$cmd" >/dev/null || fail "$label not found on PATH: $cmd"
  fi
}

write_metadata() {
  local run_dir="$1"
  {
    echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "run_date_path: $RUN_DATE_PATH"
    echo "host: $(hostname)"
    echo "cwd: $ROOT_DIR"
    echo "runtime: $RUNTIME"
    echo "model: $MODEL_PATH"
    echo "lora: $LORA_PATH"
    echo "image: ${IMAGE_PATHS[0]}"
    echo "image_paths:"
    for image_path in "${IMAGE_PATHS[@]}"; do
      echo "  - $image_path"
    done
    echo "prompt: $PROMPT"
    echo "seed: $SEED"
    echo "width: $WIDTH"
    echo "height: $HEIGHT"
    echo "steps: $STEPS"
    echo "guidance: $GUIDANCE"
    echo "true_cfg_scale: $TRUE_CFG_SCALE"
    echo "repeats: $REPEATS"
    echo "mflux_bin: $MFLUX_BIN"
    echo "shardedit_mflux_bin: $SHARDEDIT_MFLUX_BIN"
    echo "mflux_cache_limit_gb: $MFLUX_CACHE_LIMIT_GB"
    echo "eval_every_n_blocks: $EVAL_EVERY_N_BLOCKS"
    echo "probe_blocks: $PROBE_BLOCKS"
    echo "token_redundancy_blocks: $TOKEN_REDUNDANCY_BLOCKS"
    echo "token_redundancy_heatmap_dir: $TOKEN_REDUNDANCY_HEATMAP_DIR"
    echo "bridge_error_diagnose: $BRIDGE_ERROR_DIAGNOSE"
    echo "bridge_error_heatmap_dir: $BRIDGE_ERROR_HEATMAP_DIR"
    echo "selective_refill_fraction: $SELECTIVE_REFILL_FRACTION"
    echo "selective_refill_mode: $SELECTIVE_REFILL_MODE"
    echo "selective_refill_dampen: $SELECTIVE_REFILL_DAMPEN"
    echo "selective_refill_min_step: $SELECTIVE_REFILL_MIN_STEP"
    echo "cache_threshold: $CACHE_THRESHOLD"
    echo "cache_max_consecutive: $CACHE_MAX_CONSECUTIVE"
    echo "cache_warmup_steps: $CACHE_WARMUP_STEPS"
    echo "cache_back_blocks: $CACHE_BACK_BLOCKS"
    echo "cache_anchor_mode: $CACHE_ANCHOR_MODE"
    echo "cache_predictor: $CACHE_PREDICTOR"
    echo "cache_threshold_schedule: $CACHE_THRESHOLD_SCHEDULE"
    echo "cache_region_policy: $CACHE_REGION_POLICY"
    echo "cache_preset: $CACHE_PRESET"
    echo "reference_conditioning_size: $REFERENCE_CONDITIONING_SIZE"
    echo "reference_conditioning_short_side: $REFERENCE_CONDITIONING_SHORT_SIDE"
    echo "reference_conditioning_max_width: $REFERENCE_CONDITIONING_MAX_WIDTH"
    echo "reference_conditioning_max_height: $REFERENCE_CONDITIONING_MAX_HEIGHT"
    echo "residency_mode: $RESIDENCY_MODE"
    echo "residency_window_size: $RESIDENCY_WINDOW_SIZE"
    echo "release_policy: $RELEASE_POLICY"
    echo "dense_img_ff_window: $DENSE_IMG_FF_WINDOW"
    echo "dense_img_ff_cache_max_blocks: $DENSE_IMG_FF_CACHE_MAX_BLOCKS"
    echo "kquant_img_ff_window: $KQUANT_IMG_FF_WINDOW"
    echo "kquant_img_ff_cache_max_blocks: $KQUANT_IMG_FF_CACHE_MAX_BLOCKS"
    echo "kquant_img_ff_codec: $KQUANT_IMG_FF_CODEC"
    echo "lora_tensor_cache: $LORA_TENSOR_CACHE"
    echo "lora_tensor_cache_max_windows: $LORA_TENSOR_CACHE_MAX_WINDOWS"
    echo "patched_window_cache_max_windows: $PATCHED_WINDOW_CACHE_MAX_WINDOWS"
    echo "condition_token_merge: $CONDITION_TOKEN_MERGE"
    echo "condition_token_merge_stride: $CONDITION_TOKEN_MERGE_STRIDE"
    echo "condition_token_merge_start_block: $CONDITION_TOKEN_MERGE_START_BLOCK"
    echo "condition_token_merge_back_blocks: $CONDITION_TOKEN_MERGE_BACK_BLOCKS"
    echo "text_token_merge: $TEXT_TOKEN_MERGE"
    echo "text_token_merge_stride: $TEXT_TOKEN_MERGE_STRIDE"
    echo "text_token_merge_start_block: $TEXT_TOKEN_MERGE_START_BLOCK"
    echo "text_token_merge_back_blocks: $TEXT_TOKEN_MERGE_BACK_BLOCKS"
    echo "q6_linear_profile: $Q6_LINEAR_PROFILE"
    echo "shardedit_profile: $SHARDEDIT_PROFILE"
    echo "qwen_image_cli: $QWEN_IMAGE_CLI"
    echo "gpu_cache_limit: $GPU_CACHE_LIMIT"
    echo "cooldown_seconds: $COOLDOWN_SECONDS"
    echo "run_sequence_label: $RUN_SEQUENCE_LABEL"
    echo "condition_note: $CONDITION_NOTE"
    echo "thermal_note: $THERMAL_NOTE"
    echo "memory_pressure_before:"
    memory_pressure 2>/dev/null | sed 's/^/  /' || true
    echo "swapusage_before:"
    sysctl vm.swapusage 2>/dev/null | sed 's/^/  /' || true
    echo "sw_vers:"
    sw_vers 2>/dev/null | sed 's/^/  /' || true
    echo "uname: $(uname -a)"
    echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    echo "mem_bytes: $(sysctl -n hw.memsize 2>/dev/null || true)"
    echo "swift_version:"
    swift --version 2>/dev/null | sed 's/^/  /' || true
    echo "mflux_version:"
    mflux_python="$(mflux_python_path || true)"
    if [[ -n "$mflux_python" && -x "$mflux_python" ]]; then
      "$mflux_python" -c 'import importlib.metadata as m; print(m.version("mflux"))' 2>/dev/null | sed 's/^/  /' || true
    fi
  } > "$run_dir/metadata.txt"
}

mflux_python_path() {
  local resolved
  resolved="$(command -v "$MFLUX_BIN" 2>/dev/null || true)"
  [[ -n "$resolved" ]] || return 0
  local shebang
  shebang="$(head -n 1 "$resolved" 2>/dev/null || true)"
  case "$shebang" in
    '#!'*)
      shebang="${shebang#\#!}"
      echo "${shebang%% *}"
      ;;
  esac
}

run_with_capture() {
  local label="$1"
  shift

  local run_dir="$CURRENT_RUN_DIR/$label"
  mkdir -p "$run_dir"

  quote_command "$@" > "$run_dir/command.sh"
  chmod +x "$run_dir/command.sh"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "dry-run: $label"
    quote_command "$@"
    return 0
  fi

  if [[ "$COOLDOWN_SECONDS" -gt 0 ]]; then
    echo "cooldown: ${COOLDOWN_SECONDS}s before $label"
    sleep "$COOLDOWN_SECONDS"
  fi

  echo "running: $label"
  local start_epoch
  local end_epoch
  local started_at
  local ended_at
  start_epoch="$(date +%s)"
  started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  /usr/bin/time -l "$@" > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
  local status=$?
  end_epoch="$(date +%s)"
  ended_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  {
    echo "status: $status"
    echo "elapsed_seconds: $((end_epoch - start_epoch))"
    echo "started_at: $started_at"
    echo "ended_at: $ended_at"
    echo "cooldown_seconds_before_run: $COOLDOWN_SECONDS"
    echo "run_sequence_label: $RUN_SEQUENCE_LABEL"
  } > "$run_dir/result.txt"

  if [[ "$status" -ne 0 ]]; then
    echo "failed: $label (status $status); see $run_dir/stderr.log" >&2
  else
    echo "finished: $label"
  fi
  return "$status"
}

build_mflux_command() {
  local output="$1"
  local executable="$2"
  MFLUX_COMMAND=(
    "$executable"
    --model "$MODEL_PATH"
    --base-model qwen
    --image-paths "${IMAGE_PATHS[@]}"
    --prompt "$PROMPT"
    --seed "$SEED"
    --steps "$STEPS"
    --guidance "$GUIDANCE"
    --width "$WIDTH"
    --height "$HEIGHT"
    --lora-paths "$LORA_PATH"
    --lora-scales 1.0
    --output "$output"
  )
  if [[ "$LOW_RAM" -eq 1 ]]; then
    MFLUX_COMMAND+=(--low-ram)
  fi
  if [[ -n "$MFLUX_CACHE_LIMIT_GB" ]]; then
    MFLUX_COMMAND+=(--mlx-cache-limit-gb "$MFLUX_CACHE_LIMIT_GB")
  fi
}

build_swift_command() {
  local output="$1"
  SWIFT_COMMAND=(
    "$QWEN_IMAGE_CLI"
    --model "$MODEL_PATH"
    --reference-image "${IMAGE_PATHS[0]}"
    --prompt "$PROMPT"
    --seed "$SEED"
    --steps "$STEPS"
    --guidance "$GUIDANCE"
    --true-cfg-scale "$TRUE_CFG_SCALE"
    --width "$WIDTH"
    --height "$HEIGHT"
    --lora "$LORA_PATH"
    --profile
    --gpu-cache-limit "$GPU_CACHE_LIMIT"
    --clear-cache-between-stages
    --output "$output"
  )
}

apply_cache_preset
apply_cache_env_overrides

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime) RUNTIME="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --lora) LORA_PATH="$2"; shift 2 ;;
    --image)
      if [[ "$IMAGE_OPTION_SEEN" -eq 0 ]]; then
        IMAGE_PATHS=()
        IMAGE_OPTION_SEEN=1
      fi
      IMAGE_PATHS+=("$2")
      shift 2
      ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --guidance) GUIDANCE="$2"; shift 2 ;;
    --true-cfg-scale) TRUE_CFG_SCALE="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --mflux-bin) MFLUX_BIN="$2"; shift 2 ;;
    --shardedit-mflux-bin) SHARDEDIT_MFLUX_BIN="$2"; shift 2 ;;
    --mflux-cache-limit-gb) MFLUX_CACHE_LIMIT_GB="$2"; shift 2 ;;
    --eval-every-n-blocks) EVAL_EVERY_N_BLOCKS="$2"; shift 2 ;;
    --probe-blocks) PROBE_BLOCKS="$2"; shift 2 ;;
    --token-redundancy-blocks) TOKEN_REDUNDANCY_BLOCKS="$2"; shift 2 ;;
    --token-redundancy-heatmap-dir) TOKEN_REDUNDANCY_HEATMAP_DIR="$2"; shift 2 ;;
    --bridge-error-diagnose) BRIDGE_ERROR_DIAGNOSE=1; shift ;;
    --bridge-error-heatmap-dir) BRIDGE_ERROR_HEATMAP_DIR="$2"; shift 2 ;;
    --selective-refill-fraction) SELECTIVE_REFILL_FRACTION="$2"; shift 2 ;;
    --selective-refill-mode) SELECTIVE_REFILL_MODE="$2"; shift 2 ;;
    --selective-refill-dampen) SELECTIVE_REFILL_DAMPEN="$2"; shift 2 ;;
    --selective-refill-min-step) SELECTIVE_REFILL_MIN_STEP="$2"; shift 2 ;;
    --cache-threshold) CACHE_THRESHOLD="$2"; shift 2 ;;
    --cache-max-consecutive) CACHE_MAX_CONSECUTIVE="$2"; shift 2 ;;
    --cache-warmup-steps) CACHE_WARMUP_STEPS="$2"; shift 2 ;;
    --cache-back-blocks) CACHE_BACK_BLOCKS="$2"; shift 2 ;;
    --cache-anchor-mode) CACHE_ANCHOR_MODE="$2"; shift 2 ;;
    --cache-predictor) CACHE_PREDICTOR="$2"; shift 2 ;;
    --cache-threshold-schedule) CACHE_THRESHOLD_SCHEDULE="$2"; shift 2 ;;
    --cache-region-policy) CACHE_REGION_POLICY="$2"; shift 2 ;;
    --cache-preset) CACHE_PRESET="$2"; apply_cache_preset; shift 2 ;;
    --reference-conditioning-size) REFERENCE_CONDITIONING_SIZE="$2"; shift 2 ;;
    --reference-conditioning-short-side) REFERENCE_CONDITIONING_SHORT_SIDE="$2"; shift 2 ;;
    --reference-conditioning-max-width) REFERENCE_CONDITIONING_MAX_WIDTH="$2"; shift 2 ;;
    --reference-conditioning-max-height) REFERENCE_CONDITIONING_MAX_HEIGHT="$2"; shift 2 ;;
    --residency) RESIDENCY_MODE="$2"; shift 2 ;;
    --residency-window-size) RESIDENCY_WINDOW_SIZE="$2"; shift 2 ;;
    --release-policy) RELEASE_POLICY="$2"; shift 2 ;;
    --dense-img-ff-window) DENSE_IMG_FF_WINDOW=1; shift ;;
    --dense-img-ff-cache-max-blocks) DENSE_IMG_FF_CACHE_MAX_BLOCKS="$2"; shift 2 ;;
    --kquant-img-ff-window) KQUANT_IMG_FF_WINDOW=1; shift ;;
    --kquant-img-ff-cache-max-blocks) KQUANT_IMG_FF_CACHE_MAX_BLOCKS="$2"; shift 2 ;;
    --kquant-img-ff-codec) KQUANT_IMG_FF_CODEC="$2"; shift 2 ;;
    --lora-tensor-cache) LORA_TENSOR_CACHE=1; shift ;;
    --lora-tensor-cache-max-windows) LORA_TENSOR_CACHE_MAX_WINDOWS="$2"; shift 2 ;;
    --patched-window-cache-max-windows) PATCHED_WINDOW_CACHE_MAX_WINDOWS="$2"; shift 2 ;;
    --condition-token-merge) CONDITION_TOKEN_MERGE=1; shift ;;
    --condition-token-merge-stride) CONDITION_TOKEN_MERGE_STRIDE="$2"; shift 2 ;;
    --condition-token-merge-start-block) CONDITION_TOKEN_MERGE_START_BLOCK="$2"; shift 2 ;;
    --condition-token-merge-back-blocks) CONDITION_TOKEN_MERGE_BACK_BLOCKS="$2"; shift 2 ;;
    --text-token-merge) TEXT_TOKEN_MERGE=1; shift ;;
    --text-token-merge-stride) TEXT_TOKEN_MERGE_STRIDE="$2"; shift 2 ;;
    --text-token-merge-start-block) TEXT_TOKEN_MERGE_START_BLOCK="$2"; shift 2 ;;
    --text-token-merge-back-blocks) TEXT_TOKEN_MERGE_BACK_BLOCKS="$2"; shift 2 ;;
    --q6-linear-profile) Q6_LINEAR_PROFILE=1; shift ;;
    --no-shardedit-profile) SHARDEDIT_PROFILE=0; shift ;;
    --qwen-image-cli) QWEN_IMAGE_CLI="$2"; shift 2 ;;
    --gpu-cache-limit) GPU_CACHE_LIMIT="$2"; shift 2 ;;
    --cooldown-seconds) COOLDOWN_SECONDS="$2"; shift 2 ;;
    --run-sequence-label) RUN_SEQUENCE_LABEL="$2"; shift 2 ;;
    --condition-note) CONDITION_NOTE="$2"; shift 2 ;;
    --thermal-note) THERMAL_NOTE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done

case "$RUNTIME" in
  mflux|mflux-no-lowram|shardedit|swift|both) ;;
  *) fail "--runtime must be mflux, mflux-no-lowram, shardedit, swift, or both" ;;
esac

[[ "$REPEATS" =~ ^[0-9]+$ ]] || fail "--repeats must be an integer"
[[ "$REPEATS" -ge 1 ]] || fail "--repeats must be >= 1"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "--seed must be a non-negative integer"
case "$RESIDENCY_MODE" in
  none|shard|window) ;;
  *) fail "--residency must be none, shard, or window" ;;
esac
[[ "$RESIDENCY_WINDOW_SIZE" =~ ^[0-9]+$ ]] || fail "--residency-window-size must be an integer"
[[ "$RESIDENCY_WINDOW_SIZE" -ge 1 ]] || fail "--residency-window-size must be >= 1"
case "$RELEASE_POLICY" in
  window|step|none|keep-last) ;;
  *) fail "--release-policy must be window, step, none, or keep-last" ;;
esac
[[ "$COOLDOWN_SECONDS" =~ ^[0-9]+$ ]] || fail "--cooldown-seconds must be a non-negative integer"
case "$CACHE_ANCHOR_MODE" in
  residual|absolute) ;;
  *) fail "--cache-anchor-mode must be residual or absolute" ;;
esac
case "$CACHE_PREDICTOR" in
  last|linear|linear-residual|quadratic|quadratic-residual|adams-bashforth|adams-bashforth-residual) ;;
  *) fail "--cache-predictor must be last, linear, linear-residual, quadratic, quadratic-residual, adams-bashforth, or adams-bashforth-residual" ;;
esac
case "$CACHE_THRESHOLD_SCHEDULE" in
  fixed|sigma|flow-aware|flow-aware-veto) ;;
  *) fail "--cache-threshold-schedule must be fixed, sigma, flow-aware, or flow-aware-veto" ;;
esac
case "$CACHE_REGION_POLICY" in
  all|target-conservative|condition-conservative) ;;
  *) fail "--cache-region-policy must be all, target-conservative, or condition-conservative" ;;
esac
case "$REFERENCE_CONDITIONING_SIZE" in
  upstream|original|short-side|short-side-512|fit-box) ;;
  *) fail "--reference-conditioning-size must be upstream, original, short-side, short-side-512, or fit-box" ;;
esac
[[ "$REFERENCE_CONDITIONING_SHORT_SIDE" =~ ^[0-9]+$ ]] || fail "--reference-conditioning-short-side must be an integer"
[[ "$REFERENCE_CONDITIONING_SHORT_SIDE" -ge 1 ]] || fail "--reference-conditioning-short-side must be >= 1"
[[ "$REFERENCE_CONDITIONING_MAX_WIDTH" =~ ^[0-9]+$ ]] || fail "--reference-conditioning-max-width must be an integer"
[[ "$REFERENCE_CONDITIONING_MAX_WIDTH" -ge 1 ]] || fail "--reference-conditioning-max-width must be >= 1"
[[ "$REFERENCE_CONDITIONING_MAX_HEIGHT" =~ ^[0-9]+$ ]] || fail "--reference-conditioning-max-height must be an integer"
[[ "$REFERENCE_CONDITIONING_MAX_HEIGHT" -ge 1 ]] || fail "--reference-conditioning-max-height must be >= 1"
if [[ "$REFERENCE_CONDITIONING_SIZE" == "fit-box" ]]; then
  [[ "$REFERENCE_CONDITIONING_MAX_WIDTH" -ge 32 ]] || fail "fit-box --reference-conditioning-max-width must be >= 32"
  [[ "$REFERENCE_CONDITIONING_MAX_HEIGHT" -ge 32 ]] || fail "fit-box --reference-conditioning-max-height must be >= 32"
fi
case "$CONDITION_TOKEN_MERGE" in
  0|1) ;;
  *) fail "--condition-token-merge must be 0/1 when supplied through SHARDEDIT_CONDITION_TOKEN_MERGE" ;;
esac
case "$LORA_TENSOR_CACHE" in
  0|1) ;;
  *) fail "--lora-tensor-cache must be 0/1 when supplied through SHARDEDIT_LORA_TENSOR_CACHE" ;;
esac
case "$KQUANT_IMG_FF_WINDOW" in
  0|1) ;;
  *) fail "--kquant-img-ff-window must be 0/1 when supplied through SHARDEDIT_KQUANT_IMG_FF_WINDOW" ;;
esac
[[ "$KQUANT_IMG_FF_CACHE_MAX_BLOCKS" =~ ^[0-9]+$ ]] || fail "--kquant-img-ff-cache-max-blocks must be an integer"
[[ "$KQUANT_IMG_FF_CACHE_MAX_BLOCKS" -ge 1 ]] || fail "--kquant-img-ff-cache-max-blocks must be >= 1"
[[ "$LORA_TENSOR_CACHE_MAX_WINDOWS" =~ ^[0-9]+$ ]] || fail "--lora-tensor-cache-max-windows must be an integer"
[[ "$LORA_TENSOR_CACHE_MAX_WINDOWS" -ge 1 ]] || fail "--lora-tensor-cache-max-windows must be >= 1"
[[ "$PATCHED_WINDOW_CACHE_MAX_WINDOWS" =~ ^[0-9]+$ ]] || fail "--patched-window-cache-max-windows must be a non-negative integer"
if [[ "$RESIDENCY_MODE" == "none" && "$LORA_TENSOR_CACHE" -eq 1 ]]; then
  fail "--lora-tensor-cache requires --residency shard or window"
fi
if [[ "$RESIDENCY_MODE" == "none" && "$KQUANT_IMG_FF_WINDOW" -eq 1 ]]; then
  fail "--kquant-img-ff-window requires --residency shard or window"
fi
if [[ "$DENSE_IMG_FF_WINDOW" -eq 1 && "$KQUANT_IMG_FF_WINDOW" -eq 1 ]]; then
  fail "--dense-img-ff-window and --kquant-img-ff-window are mutually exclusive"
fi
if [[ "$RESIDENCY_MODE" == "none" && "$PATCHED_WINDOW_CACHE_MAX_WINDOWS" -gt 0 ]]; then
  fail "--patched-window-cache-max-windows requires --residency shard or window"
fi
[[ "$CONDITION_TOKEN_MERGE_STRIDE" =~ ^[0-9]+$ ]] || fail "--condition-token-merge-stride must be an integer"
[[ "$CONDITION_TOKEN_MERGE_STRIDE" -ge 2 ]] || fail "--condition-token-merge-stride must be >= 2"
[[ "$CONDITION_TOKEN_MERGE_START_BLOCK" =~ ^[0-9]+$ ]] || fail "--condition-token-merge-start-block must be an integer"
[[ "$CONDITION_TOKEN_MERGE_START_BLOCK" -ge 1 ]] || fail "--condition-token-merge-start-block must be >= 1"
[[ "$CONDITION_TOKEN_MERGE_BACK_BLOCKS" =~ ^[0-9]+$ ]] || fail "--condition-token-merge-back-blocks must be a non-negative integer"
case "$TEXT_TOKEN_MERGE" in
  0|1) ;;
  *) fail "--text-token-merge must be 0/1 when supplied through SHARDEDIT_TEXT_TOKEN_MERGE" ;;
esac
[[ "$TEXT_TOKEN_MERGE_STRIDE" =~ ^[0-9]+$ ]] || fail "--text-token-merge-stride must be an integer"
[[ "$TEXT_TOKEN_MERGE_STRIDE" -ge 2 ]] || fail "--text-token-merge-stride must be >= 2"
[[ "$TEXT_TOKEN_MERGE_START_BLOCK" =~ ^[0-9]+$ ]] || fail "--text-token-merge-start-block must be an integer"
[[ "$TEXT_TOKEN_MERGE_START_BLOCK" -ge 1 ]] || fail "--text-token-merge-start-block must be >= 1"
[[ "$TEXT_TOKEN_MERGE_BACK_BLOCKS" =~ ^[0-9]+$ ]] || fail "--text-token-merge-back-blocks must be a non-negative integer"
case "$Q6_LINEAR_PROFILE" in
  0|1) ;;
  *) fail "--q6-linear-profile must be 0/1 when supplied through SHARDEDIT_Q6_LINEAR_PROFILE" ;;
esac

need_file "model" "$MODEL_PATH"
need_file "LoRA" "$LORA_PATH"
for image_path in "${IMAGE_PATHS[@]}"; do
  need_file "image" "$image_path"
done

RUN_DATE_PATH="$(date '+%Y-%m-%d')"
timestamp="$(date '+%Y%m%d-%H%M%S')"
CURRENT_RUN_DIR="$OUTPUT_ROOT/$RUN_DATE_PATH/$timestamp"
suffix=2
while [[ -e "$CURRENT_RUN_DIR" ]]; do
  CURRENT_RUN_DIR="$OUTPUT_ROOT/$RUN_DATE_PATH/${timestamp}-${suffix}"
  suffix=$((suffix + 1))
done
mkdir -p "$CURRENT_RUN_DIR"
write_metadata "$CURRENT_RUN_DIR"

echo "benchmark directory: $CURRENT_RUN_DIR"

overall_status=0
for ((i = 1; i <= REPEATS; i++)); do
  if [[ "$RUNTIME" == "mflux" || "$RUNTIME" == "mflux-no-lowram" || "$RUNTIME" == "both" ]]; then
    need_executable_command "mflux" "$MFLUX_BIN"
    LOW_RAM=1
    [[ "$RUNTIME" == "mflux-no-lowram" ]] && LOW_RAM=0
    mflux_label="mflux"
    [[ "$LOW_RAM" -eq 1 ]] && mflux_label="mflux-lowram"
    output="$CURRENT_RUN_DIR/${mflux_label}-${i}.png"
    build_mflux_command "$output" "$MFLUX_BIN"
    run_with_capture "${mflux_label}-${i}" "${MFLUX_COMMAND[@]}" || overall_status=$?
  fi

  if [[ "$RUNTIME" == "shardedit" ]]; then
    need_executable_command "shardedit-mflux-edit" "$SHARDEDIT_MFLUX_BIN"
    LOW_RAM=1
    output="$CURRENT_RUN_DIR/shardedit-${i}.png"
    build_mflux_command "$output" "$SHARDEDIT_MFLUX_BIN"
    if [[ "$EVAL_EVERY_N_BLOCKS" -gt 0 ]]; then
      MFLUX_COMMAND+=(--shardedit-eval-every-n-blocks "$EVAL_EVERY_N_BLOCKS")
    fi
    if [[ -n "$PROBE_BLOCKS" ]]; then
      MFLUX_COMMAND+=(--shardedit-probe-blocks "$PROBE_BLOCKS")
    fi
    if [[ -n "$TOKEN_REDUNDANCY_BLOCKS" ]]; then
      MFLUX_COMMAND+=(--shardedit-token-redundancy-blocks "$TOKEN_REDUNDANCY_BLOCKS")
    fi
    if [[ -n "$TOKEN_REDUNDANCY_HEATMAP_DIR" ]]; then
      MFLUX_COMMAND+=(--shardedit-token-redundancy-heatmap-dir "$TOKEN_REDUNDANCY_HEATMAP_DIR")
    fi
    if [[ "$BRIDGE_ERROR_DIAGNOSE" == "1" ]]; then
      MFLUX_COMMAND+=(--shardedit-bridge-error-diagnose)
    fi
    if [[ -n "$BRIDGE_ERROR_HEATMAP_DIR" ]]; then
      MFLUX_COMMAND+=(--shardedit-bridge-error-heatmap-dir "$BRIDGE_ERROR_HEATMAP_DIR")
    fi
    if [[ "$SELECTIVE_REFILL_FRACTION" != "0" && "$SELECTIVE_REFILL_FRACTION" != "0.0" ]]; then
      MFLUX_COMMAND+=(--shardedit-selective-refill-fraction "$SELECTIVE_REFILL_FRACTION")
      MFLUX_COMMAND+=(--shardedit-selective-refill-mode "$SELECTIVE_REFILL_MODE")
      MFLUX_COMMAND+=(--shardedit-selective-refill-dampen "$SELECTIVE_REFILL_DAMPEN")
      MFLUX_COMMAND+=(--shardedit-selective-refill-min-step "$SELECTIVE_REFILL_MIN_STEP")
    fi
    MFLUX_COMMAND+=(
      --shardedit-cache-threshold "$CACHE_THRESHOLD"
      --shardedit-cache-max-consecutive "$CACHE_MAX_CONSECUTIVE"
      --shardedit-cache-warmup-steps "$CACHE_WARMUP_STEPS"
      --shardedit-cache-back-blocks "$CACHE_BACK_BLOCKS"
      --shardedit-cache-anchor-mode "$CACHE_ANCHOR_MODE"
      --shardedit-cache-predictor "$CACHE_PREDICTOR"
      --shardedit-cache-threshold-schedule "$CACHE_THRESHOLD_SCHEDULE"
      --shardedit-cache-region-policy "$CACHE_REGION_POLICY"
      --shardedit-reference-conditioning-size "$REFERENCE_CONDITIONING_SIZE"
      --shardedit-reference-conditioning-short-side "$REFERENCE_CONDITIONING_SHORT_SIDE"
      --shardedit-reference-conditioning-max-width "$REFERENCE_CONDITIONING_MAX_WIDTH"
      --shardedit-reference-conditioning-max-height "$REFERENCE_CONDITIONING_MAX_HEIGHT"
    )
    if [[ "$RESIDENCY_MODE" != "none" ]]; then
      MFLUX_COMMAND+=(--shardedit-residency "$RESIDENCY_MODE")
      MFLUX_COMMAND+=(--shardedit-release-policy "$RELEASE_POLICY")
      if [[ "$RESIDENCY_MODE" == "window" ]]; then
        MFLUX_COMMAND+=(--shardedit-residency-window-size "$RESIDENCY_WINDOW_SIZE")
      fi
      if [[ "$DENSE_IMG_FF_WINDOW" -eq 1 ]]; then
        MFLUX_COMMAND+=(--shardedit-dense-img-ff-window)
        MFLUX_COMMAND+=(--shardedit-dense-img-ff-cache-max-blocks "$DENSE_IMG_FF_CACHE_MAX_BLOCKS")
      fi
      if [[ "$KQUANT_IMG_FF_WINDOW" -eq 1 ]]; then
        MFLUX_COMMAND+=(--shardedit-kquant-img-ff-window)
        MFLUX_COMMAND+=(--shardedit-kquant-img-ff-cache-max-blocks "$KQUANT_IMG_FF_CACHE_MAX_BLOCKS")
        MFLUX_COMMAND+=(--shardedit-kquant-img-ff-codec "$KQUANT_IMG_FF_CODEC")
      fi
      if [[ "$LORA_TENSOR_CACHE" -eq 1 ]]; then
        MFLUX_COMMAND+=(--shardedit-lora-tensor-cache)
        MFLUX_COMMAND+=(--shardedit-lora-tensor-cache-max-windows "$LORA_TENSOR_CACHE_MAX_WINDOWS")
      fi
      if [[ "$PATCHED_WINDOW_CACHE_MAX_WINDOWS" -gt 0 ]]; then
        MFLUX_COMMAND+=(--shardedit-patched-window-cache-max-windows "$PATCHED_WINDOW_CACHE_MAX_WINDOWS")
      fi
    fi
    if [[ "$DENSE_IMG_FF_WINDOW" -eq 1 && "$RESIDENCY_MODE" == "none" ]]; then
      fail "--dense-img-ff-window requires --residency shard or window"
    fi
    if [[ "$KQUANT_IMG_FF_WINDOW" -eq 1 && "$RESIDENCY_MODE" == "none" ]]; then
      fail "--kquant-img-ff-window requires --residency shard or window"
    fi
    if [[ "$CONDITION_TOKEN_MERGE" -eq 1 ]]; then
      MFLUX_COMMAND+=(
        --shardedit-condition-token-merge
        --shardedit-condition-token-merge-stride "$CONDITION_TOKEN_MERGE_STRIDE"
        --shardedit-condition-token-merge-start-block "$CONDITION_TOKEN_MERGE_START_BLOCK"
        --shardedit-condition-token-merge-back-blocks "$CONDITION_TOKEN_MERGE_BACK_BLOCKS"
      )
    fi
    if [[ "$TEXT_TOKEN_MERGE" -eq 1 ]]; then
      MFLUX_COMMAND+=(
        --shardedit-text-token-merge
        --shardedit-text-token-merge-stride "$TEXT_TOKEN_MERGE_STRIDE"
        --shardedit-text-token-merge-start-block "$TEXT_TOKEN_MERGE_START_BLOCK"
        --shardedit-text-token-merge-back-blocks "$TEXT_TOKEN_MERGE_BACK_BLOCKS"
      )
    fi
    if [[ "$SHARDEDIT_PROFILE" -eq 1 ]]; then
      MFLUX_COMMAND+=(--shardedit-profile)
    fi
    if [[ "$Q6_LINEAR_PROFILE" -eq 1 ]]; then
      MFLUX_COMMAND+=(--shardedit-q6-linear-profile)
    fi
    run_with_capture "shardedit-${i}" "${MFLUX_COMMAND[@]}" || overall_status=$?
  fi

  if [[ "$RUNTIME" == "swift" || "$RUNTIME" == "both" ]]; then
    need_executable_command "QwenImageCLI" "$QWEN_IMAGE_CLI"
    output="$CURRENT_RUN_DIR/swift-${i}.png"
    build_swift_command "$output"
    run_with_capture "swift-${i}" "${SWIFT_COMMAND[@]}" || overall_status=$?
  fi
done

echo "summary command:"
echo "  python3 tools/summarize_benchmarks.py $CURRENT_RUN_DIR"

exit "$overall_status"
