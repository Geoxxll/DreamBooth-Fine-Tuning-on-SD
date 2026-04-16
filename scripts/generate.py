#!/usr/bin/env python
"""Load a trained LoRA folder and generate images for prompts (e.g. for reports / submission)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DreamBooth LoRA inference (SD 1.5).")
    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model id or path (must match training).",
    )
    p.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Directory containing pytorch_lora_weights.safetensors from training output_dir.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="samples",
        help="Where to save PNGs (default: samples/).",
    )
    p.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Text file with one prompt per line. If omitted, use --prompt.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt (used if --prompts_file not set).",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds; one image per (prompt, seed) pair.",
    )
    # Slightly softer defaults than 30 / 7.5 — finetuned LoRAs often look cleaner at ~7.0 CFG.
    p.add_argument("--num_inference_steps", type=int, default=35)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument(
        "--weight_name",
        type=str,
        default="pytorch_lora_weights.safetensors",
        help="LoRA filename inside lora_path (diffusers DreamBooth LoRA default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        raise SystemExit("Provide --prompts_file or --prompt.")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.load_lora_weights(args.lora_path, weight_name=args.weight_name)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    idx = 0
    for pi, prompt in enumerate(prompts):
        for si, seed in enumerate(seeds):
            gen = torch.Generator(device=pipe.device).manual_seed(seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=gen,
            ).images[0]
            name = f"gen_p{pi:03d}_s{seed}_{idx:04d}.png"
            path = out / name
            image.save(path)
            print(f"Wrote {path}")
            idx += 1


if __name__ == "__main__":
    main()
