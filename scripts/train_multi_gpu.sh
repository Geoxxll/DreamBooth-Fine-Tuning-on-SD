#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_ARGS_FILE="${TRAIN_ARGS_FILE:-$REPO_ROOT/configs/train_args.sh}"
if [[ ! -f "$TRAIN_ARGS_FILE" ]]; then
  echo "Missing $TRAIN_ARGS_FILE — copy configs/train_args.example.sh to configs/train_args.sh and edit."
  exit 1
fi
# shellcheck source=/dev/null
source "$TRAIN_ARGS_FILE"

NUM_PROCESSES="${NUM_PROCESSES:-1}"

PRIOR_FLAGS=()
if [[ "${WITH_PRIOR_PRESERVATION:-false}" == "true" ]]; then
  PRIOR_FLAGS+=(--with_prior_preservation --num_class_images "${NUM_CLASS_IMAGES:-100}")
  if [[ -n "${CLASS_DATA_DIR:-}" ]]; then
    PRIOR_FLAGS+=(--class_data_dir "$REPO_ROOT/$CLASS_DATA_DIR")
  fi
fi

CLASS_FLAGS=()
if [[ -n "${CLASS_PROMPT:-}" ]]; then
  CLASS_FLAGS+=(--class_prompt "$CLASS_PROMPT")
fi

exec accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  "$REPO_ROOT/scripts/train_dreambooth_lora.py" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --instance_data_dir "$REPO_ROOT/$INSTANCE_DATA_DIR" \
  --instance_prompt "$INSTANCE_PROMPT" \
  "${CLASS_FLAGS[@]}" \
  "${PRIOR_FLAGS[@]}" \
  --output_dir "$REPO_ROOT/$OUTPUT_DIR" \
  --resolution "$RESOLUTION" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler "$LR_SCHEDULER" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --checkpointing_steps "$CHECKPOINTING_STEPS" \
  --mixed_precision "$MIXED_PRECISION" \
  --gradient_checkpointing \
  --allow_tf32 \
  --seed "$SEED" \
  --rank "$LORA_RANK" \
  --validation_prompt "$VALIDATION_PROMPT" \
  --num_validation_images "$NUM_VALIDATION_IMAGES" \
  --validation_epochs "$VALIDATION_EPOCHS" \
  --report_to tensorboard \
  "$@"
