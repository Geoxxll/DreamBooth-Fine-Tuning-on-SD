# DreamBooth LoRA (SD 1.5) — course project layout

This repo follows a **Hugging Face diffusers + accelerate** workflow: **data → LoRA fine-tuning → generation → report**. Defaults assume **one 12GB-class GPU** (e.g. RTX 5070) using **LoRA**, not full-model DreamBooth; you can raise `NUM_PROCESSES` for multi-GPU.

## Layout

| Path | Purpose |
|------|---------|
| `data/instance/` | 3–5 training images (**gitignored**; do not push personal photos) |
| `configs/train_args.example.sh` | Copy to `configs/train_args.sh` and edit prompts/paths |
| `configs/dreambooth_lora.example.yaml` | Human-readable hyperparameter reference |
| `scripts/train_dreambooth_lora.py` | Vendored from diffusers `v0.31.0` (see `scripts/VENDORED.md`) |
| `scripts/train_multi_gpu.sh` | `accelerate launch` wrapper (single- or multi-GPU via `NUM_PROCESSES`) |
| `scripts/generate.py` | Load LoRA and save PNGs to `samples/` |
| `outputs/` | Checkpoints & LoRA weights (**contents gitignored**) |
| `samples/` | Curated outputs for your PDF report |
| `report/` | Report source (optional) |

## Environment

1. **PyTorch with CUDA** — install the wheel that matches your server’s driver/CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) *before* other deps, e.g.:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Hugging Face token** (optional for public `runwayml/stable-diffusion-v1-5`):

   ```bash
   cp .env.example .env
   # set HF_TOKEN=... if needed; or: huggingface-cli login
   ```

4. **Accelerate** — configure once per machine; see [scripts/accelerate_config_hint.md](scripts/accelerate_config_hint.md).

## Prepare data

Add **3–5** images of one subject to `data/instance/` (see [data/README.md](data/README.md)).

## Training

1. Copy and edit training variables:

   ```bash
   cp configs/train_args.example.sh configs/train_args.sh
   # edit INSTANCE_PROMPT / OUTPUT_DIR / steps / learning rate, etc.
   ```

2. Launch (`configs/train_args.example.sh` sets `NUM_PROCESSES=1` for **one GPU**; set `NUM_PROCESSES=2` and re-run `accelerate config` if you use two cards):

   ```bash
   chmod +x scripts/train_multi_gpu.sh
   ./scripts/train_multi_gpu.sh
   ```

   **RTX 5070 (12GB) checklist:** `accelerate config` → single GPU (or multi-GPU with matching process count); training uses `fp16`, `gradient_checkpointing`, `--allow_tf32`, `TRAIN_BATCH_SIZE=1` per device — typical for SD1.5 LoRA on this hardware.

   Default `train_args` use **500 steps** and **learning rate `5e-5`** so a handful of instance photos is less likely to overfit (mushy limbs, blown-out eyes). If the subject is still underfit, raise steps slightly or try `7e-5`; if it still degrades, lower steps or LR.

Logs default to TensorBoard under your `OUTPUT_DIR`. LoRA weights are written to `OUTPUT_DIR` (e.g. `pytorch_lora_weights.safetensors`).

**Reproducibility:** record in your report: base model id, identifier prompt, seed, `MAX_TRAIN_STEPS`, batch settings, and `NUM_PROCESSES`.

## Generation

Example (after training finishes):

```bash
python scripts/generate.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --lora_path outputs/lora-run1 \
  --output_dir samples \
  --prompts_file configs/prompts.example.txt \
  --seeds 42,43,44
```

Or a single prompt:

```bash
python scripts/generate.py \
  --lora_path outputs/lora-run1 \
  --prompt "a photo of sks dog in a coffee shop" \
  --seeds 42
```

## GitHub / server workflow

Clone the repo on the server, install PyTorch + `requirements.txt`, copy `configs/train_args.sh`, place images in `data/instance/`, run `./scripts/train_multi_gpu.sh`, then `scripts/generate.py`. Zip training images and final notebook/script separately for course submission as required.

## License

The vendored `scripts/train_dreambooth_lora.py` is **Apache 2.0** (Hugging Face). Your own data, configs, and report remain yours.
