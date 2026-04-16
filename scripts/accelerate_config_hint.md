# Accelerate (multi-GPU) setup

Run once per machine (interactive or non-interactive):

```bash
accelerate config
```

Suggested answers for **2× GPU (e.g. RTX 4070)** training:

- Compute environment: **This machine**
- Distributed: **multi-GPU**
- Number of machines: **1**
- Number of processes: **2** (or match visible GPUs)
- GPU ids: **all** (default)
- Mixed precision: **fp16** (or bf16 if supported and preferred)
- Dynamo: **no** (unless you know you want it)

To generate a default config without prompts (then edit `~/.cache/huggingface/accelerate/default_config.yaml` if needed):

```bash
accelerate config default
```

Training uses `accelerate launch` from `scripts/train_multi_gpu.sh`, which passes `--multi_gpu` and `--num_processes` explicitly; your saved config should still be consistent (especially mixed precision).
