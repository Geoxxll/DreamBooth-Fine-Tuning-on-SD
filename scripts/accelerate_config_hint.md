# Accelerate setup

Run once per machine (interactive or non-interactive):

```bash
accelerate config
```

Suggested answers for **1× GPU (e.g. RTX 5070, 12GB)** — matches the repo default `NUM_PROCESSES=1`:

- Compute environment: **This machine**
- Distributed: **no** (single GPU)
- Mixed precision: **fp16** (or bf16 if supported and preferred)
- Dynamo: **no** (unless you know you want it)

For **2× GPU** (e.g. two 12GB cards), use **multi-GPU**, **1** machine, **2** processes (or match visible GPUs), **fp16**, same as above.


To generate a default config without prompts (then edit `~/.cache/huggingface/accelerate/default_config.yaml` if needed):

```bash
accelerate config default
```

Training uses `accelerate launch` from `scripts/train_multi_gpu.sh` with `--num_processes` set from `NUM_PROCESSES`; your saved Accelerate config should still be consistent (especially mixed precision).
