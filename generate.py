"""
generate.py
───────────
Generate per-scene AI video clips for the MekongTunnel product reel.

Pipeline per scene:
  1. Load video diffusion model on assigned GPU
  2. Run inference to produce raw frames
  3. Composite brand overlays (text, terminal, browser chrome) via overlays/renderer.py
  4. Encode frames → scene_XX.mp4 at target fps

Run with --parallel (default) to dispatch scenes across all gpu_ids simultaneously.
"""

from __future__ import annotations

import os
import sys
import math
import logging
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import click
import yaml
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/generate.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─── Config / scene loaders ──────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_scenes(scenes_path: str) -> list[dict]:
    with open(scenes_path) as f:
        data = yaml.safe_load(f)
    return data["scenes"]


# ─── Pipeline constructors ────────────────────────────────────────────────────

def _build_animatediff(cfg: dict, gpu_id: int):
    import torch
    from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

    device = f"cuda:{gpu_id}"
    adapter = MotionAdapter.from_pretrained(
        cfg["animatediff"]["motion_adapter"],
        torch_dtype=torch.float16,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        cfg["animatediff"]["base_model"],
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if cfg["model"].get("offload_to_cpu"):
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    return pipe


def _build_wan(cfg: dict, gpu_id: int):
    import torch
    from diffusers import WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler

    device = f"cuda:{gpu_id}"
    pipe = WanPipeline.from_pretrained(
        cfg["wan"]["model_id"],
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if cfg["model"].get("offload_to_cpu"):
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe = pipe.to(device)
    return pipe


def _build_svd(cfg: dict, gpu_id: int):
    import torch
    from diffusers import StableVideoDiffusionPipeline

    device = f"cuda:{gpu_id}"
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        cfg["svd"]["model_id"],
        torch_dtype=torch.float16,
        variant="fp16",
    )
    if cfg["model"].get("offload_to_cpu"):
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe = pipe.to(device)
    return pipe


# ─── Frame generation wrappers ────────────────────────────────────────────────

def _gen_animatediff(pipe, scene: dict, cfg: dict) -> list:
    import torch
    w, h = cfg["output"]["width"], cfg["output"]["height"]
    ac = cfg["animatediff"]
    gen = torch.Generator().manual_seed(cfg["generation"]["seed"] + scene.get("_index", 0))
    result = pipe(
        prompt=scene["prompt"],
        negative_prompt=scene.get("negative_prompt") or cfg["generation"]["negative_prompt"],
        width=w, height=h,
        num_frames=ac["num_frames"],
        num_inference_steps=ac["inference_steps"],
        guidance_scale=ac["guidance_scale"],
        generator=gen,
    )
    return result.frames[0]


def _gen_wan(pipe, scene: dict, cfg: dict) -> list:
    import torch
    w, h = cfg["output"]["width"], cfg["output"]["height"]
    wc = cfg["wan"]
    gen = torch.Generator().manual_seed(cfg["generation"]["seed"] + scene.get("_index", 0))
    result = pipe(
        prompt=scene["prompt"],
        negative_prompt=scene.get("negative_prompt") or cfg["generation"]["negative_prompt"],
        width=w, height=h,
        num_frames=wc["num_frames"],
        num_inference_steps=wc["inference_steps"],
        guidance_scale=wc["guidance_scale"],
        generator=gen,
    )
    return result.frames[0]


def _gen_svd(pipe, scene: dict, cfg: dict) -> list:
    """SVD is image→video; we use a brand-coloured seed frame."""
    import torch
    from PIL import Image

    w, h = cfg["output"]["width"], cfg["output"]["height"]
    sc = cfg["svd"]
    gen = torch.Generator().manual_seed(cfg["generation"]["seed"] + scene.get("_index", 0))

    # Warm beige seed frame so SVD stays on-brand
    bg_color = tuple(int(cfg["brand"]["bg_primary"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    seed_image = Image.new("RGB", (w, h), bg_color)

    result = pipe(
        image=seed_image,
        width=w, height=h,
        num_frames=sc["num_frames"],
        motion_bucket_id=sc["motion_bucket_id"],
        noise_aug_strength=sc["noise_aug_strength"],
        generator=gen,
    )
    return result.frames[0]


# ─── Frames → MP4 ────────────────────────────────────────────────────────────

def frames_to_video(frames: list, out_path: str, fps: int, duration_sec: float):
    """Loop/trim frames to match target duration, then write MP4."""
    import imageio

    target = int(fps * duration_sec)
    if len(frames) < target:
        reps = math.ceil(target / len(frames))
        frames = (frames * reps)[:target]
    else:
        frames = frames[:target]

    np_frames = [np.array(f.convert("RGB") if hasattr(f, "convert") else f) for f in frames]
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=9,
                                macro_block_size=None)
    for frame in np_frames:
        writer.append_data(frame)
    writer.close()
    log.info(f"Saved {out_path}  ({len(np_frames)} frames @ {fps}fps)")


# ─── Scene worker (subprocess entry point) ───────────────────────────────────

def _scene_worker(args: tuple) -> tuple[str, str]:
    """Runs in its own process to isolate CUDA contexts per GPU."""
    scene, cfg, gpu_id, out_dir = args
    scene_id = scene["id"]
    out_path = str(Path(out_dir) / f"{scene_id}.mp4")

    try:
        # Late import so each subprocess initialises its own CUDA context
        from overlays.renderer import render_scene_overlay

        model = cfg["model"]["name"]
        log.info(f"[{scene_id}] GPU {gpu_id}  model={model}")

        builders = {"animatediff": _build_animatediff, "wan": _build_wan, "svd": _build_svd}
        generators = {"animatediff": _gen_animatediff, "wan": _gen_wan, "svd": _gen_svd}

        if model not in builders:
            raise ValueError(f"Unknown model: {model!r}. Choose: animatediff | wan | svd")

        pipe = builders[model](cfg, gpu_id)
        raw_frames = generators[model](pipe, scene, cfg)

        fps = cfg["output"]["fps"]
        total_frames = len(raw_frames)

        # Composite brand overlays frame-by-frame
        processed = [
            render_scene_overlay(f, scene, cfg, i, total_frames)
            for i, f in enumerate(raw_frames)
        ]

        frames_to_video(processed, out_path, fps, scene["duration"])
        log.info(f"[{scene_id}] ✓ {out_path}")
        return scene_id, out_path

    except Exception:
        log.error(f"[{scene_id}] FAILED:\n{traceback.format_exc()}")
        raise


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--config", default="config.yaml", show_default=True)
@click.option("--scenes", default=None,
              help="Comma-separated scene IDs to generate (default: all)")
@click.option("--parallel/--sequential", default=True, show_default=True,
              help="Use multi-GPU parallel generation")
@click.option("--dry-run", is_flag=True, help="Print plan and exit without generating")
@click.option("--steps", default=None, type=int,
              help="Override inference steps (e.g. 10 for quiet/fast mode)")
@click.option("--gpus", default=None,
              help="Comma-separated GPU indices to use, e.g. 0 or 0,1")
def main(config: str, scenes: Optional[str], parallel: bool, dry_run: bool,
         steps: Optional[int], gpus: Optional[str]):
    """Generate per-scene AI video clips for the MekongTunnel reel."""
    cfg = load_config(config)
    all_scenes = load_scenes(cfg["scenes_file"])

    # CLI overrides
    if steps is not None:
        for model_key in ("animatediff", "wan", "svd"):
            if model_key in cfg:
                step_key = "inference_steps" if model_key != "svd" else "num_frames"
                if "inference_steps" in cfg[model_key]:
                    cfg[model_key]["inference_steps"] = steps
        console.print(f"[yellow]Steps override: {steps}[/yellow]")

    if gpus is not None:
        cfg["model"]["gpu_ids"] = [int(g.strip()) for g in gpus.split(",")]
        console.print(f"[yellow]GPU override: {cfg['model']['gpu_ids']}[/yellow]")

    if scenes:
        wanted = {s.strip() for s in scenes.split(",")}
        all_scenes = [s for s in all_scenes if s["id"] in wanted]

    for i, s in enumerate(all_scenes):
        s["_index"] = i

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if dry_run:
        console.print("[bold cyan]Dry run — scene plan:[/bold cyan]")
        total = sum(s["duration"] for s in all_scenes)
        for s in all_scenes:
            ov = s.get("overlay", {}).get("type", "none")
            console.print(f"  {s['id']:12s}  {s['duration']}s  overlay={ov}")
        console.print(f"\n  Total: {total}s across {len(all_scenes)} scenes")
        return

    gpu_ids = cfg["model"]["gpu_ids"]
    console.print(f"[bold green]Generating {len(all_scenes)} scenes on {len(gpu_ids)} GPU(s)[/bold green]")

    tasks = [
        (scene, cfg, gpu_ids[i % len(gpu_ids)], out_dir)
        for i, scene in enumerate(all_scenes)
    ]

    if parallel and len(gpu_ids) > 1:
        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = {pool.submit(_scene_worker, t): t[0]["id"] for t in tasks}
            with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                          BarColumn(), TimeElapsedColumn()) as progress:
                bar = progress.add_task("Generating clips…", total=len(futures))
                for future in as_completed(futures):
                    sid = futures[future]
                    try:
                        _, out_path = future.result()
                        console.print(f"  [green]✓[/green] {sid} → {out_path}")
                    except Exception as exc:
                        console.print(f"  [red]✗[/red] {sid}: {exc}")
                    finally:
                        progress.advance(bar)
    else:
        for task in tasks:
            console.print(f"  {task[0]['id']}  GPU {task[2]}")
            _scene_worker(task)

    console.print("\n[bold green]All clips done. Run:[/bold green]  python stitch_video.py")


if __name__ == "__main__":
    main()
