#!/usr/bin/env bash
# Copy to `configs/train_args.sh` and customize (train_args.sh is gitignored).
#
# 2× RTX 4070 (12GB each): defaults below use fp16, batch 1/GPU, LoRA rank 4 — typical fit for SD1.5 LoRA.
# Single GPU: set NUM_PROCESSES=1 before running train_multi_gpu.sh

export NUM_PROCESSES=2

export PRETRAINED_MODEL_NAME_OR_PATH="runwayml/stable-diffusion-v1-5"

# DreamBooth prompts (identifier token + class)
export INSTANCE_PROMPT="a photo of sks dog"
export CLASS_PROMPT="a photo of a dog"

# Paths (relative to repo root)
export INSTANCE_DATA_DIR="data/instance"
export OUTPUT_DIR="outputs/lora-run1"

export RESOLUTION=512
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_TRAIN_STEPS=800
export LEARNING_RATE=1e-4
export LR_SCHEDULER="constant"
export LR_WARMUP_STEPS=0
export MIXED_PRECISION="fp16"
export SEED=42
export CHECKPOINTING_STEPS=200

export LORA_RANK=4

export VALIDATION_PROMPT="a photo of sks dog in a snowy mountain"
export NUM_VALIDATION_IMAGES=4
export VALIDATION_EPOCHS=50

# Set true to add prior preservation (requires class images / generation — see diffusers docs)
export WITH_PRIOR_PRESERVATION=false
export CLASS_DATA_DIR=""
export NUM_CLASS_IMAGES=100
