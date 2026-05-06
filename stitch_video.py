"""
stitch_video.py
───────────────
Assemble per-scene MP4 clips into one final vertical reel.

Features:
- Reads scene order from prompts/scenes.yaml
- Applies crossfade transitions between clips
- Adds a looping background music track (optional)
- Burns global caption pass-through (already on clips from generate.py)
- Outputs final MP4 to output/mekongtunnel_reel.mp4
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import click
import yaml
import numpy as np
from rich.console import Console

console = Console()

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/stitch.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─── Config helpers ──────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_scenes(scenes_path: str) -> list[dict]:
    with open(scenes_path) as f:
        data = yaml.safe_load(f)
    return data["scenes"]


# ─── Transition helpers (pure numpy / imageio) ───────────────────────────────

def crossfade(frames_a: list[np.ndarray], frames_b: list[np.ndarray], n_frames: int) -> list[np.ndarray]:
    """Linear crossfade between the tail of clip A and the head of clip B."""
    fade_out = frames_a[-n_frames:] if len(frames_a) >= n_frames else frames_a
    fade_in  = frames_b[:n_frames]  if len(frames_b) >= n_frames else frames_b
    length = min(len(fade_out), len(fade_in))
    result = []
    for i in range(length):
        alpha = i / max(length - 1, 1)
        blended = (fade_out[i] * (1 - alpha) + fade_in[i] * alpha).astype(np.uint8)
        result.append(blended)
    return result


def read_clip_frames(mp4_path: str) -> list[np.ndarray]:
    """Read all frames from an MP4 as numpy arrays (RGB)."""
    import imageio
    reader = imageio.get_reader(mp4_path)
    frames = [np.array(frame) for frame in reader]
    reader.close()
    return frames


# ─── Main assembly ───────────────────────────────────────────────────────────

def assemble(
    scene_clips: list[tuple[str, str]],  # [(scene_id, mp4_path), ...]
    output_path: str,
    fps: int,
    transition_frames: int = 8,
    music_path: Optional[str] = None,
):
    """
    Stitch clips with optional crossfades.

    Args:
        scene_clips:       ordered list of (scene_id, mp4_path)
        output_path:       final output file
        fps:               frames per second
        transition_frames: number of overlapping frames for crossfade (0 = hard cut)
        music_path:        optional background audio file
    """
    import imageio

    console.print(f"[bold cyan]Stitching {len(scene_clips)} clips → {output_path}[/bold cyan]")

    all_frames: list[np.ndarray] = []
    prev_frames: list[np.ndarray] = []

    for i, (scene_id, mp4_path) in enumerate(scene_clips):
        if not Path(mp4_path).exists():
            console.print(f"  [red]MISSING[/red] {mp4_path} — skipping")
            continue

        log.info(f"Reading {mp4_path}")
        curr_frames = read_clip_frames(mp4_path)
        console.print(f"  [green]+[/green] {scene_id}: {len(curr_frames)} frames")

        if prev_frames and transition_frames > 0:
            # Replace last N frames of assembled video with the crossfade blend
            all_frames = all_frames[:-transition_frames]
            blend = crossfade(prev_frames, curr_frames, transition_frames)
            all_frames.extend(blend)
            # Add the rest of the current clip (after the fade region)
            all_frames.extend(curr_frames[transition_frames:])
        else:
            all_frames.extend(curr_frames)

        prev_frames = curr_frames

    if not all_frames:
        console.print("[red]No frames collected — aborting.[/red]")
        sys.exit(1)

    total_sec = len(all_frames) / fps
    console.print(f"\n  Total frames : {len(all_frames)}")
    console.print(f"  Duration     : {total_sec:.1f}s @ {fps}fps")
    console.print(f"  Writing      : {output_path}")

    os.makedirs(Path(output_path).parent, exist_ok=True)

    if music_path and Path(music_path).exists():
        _write_with_audio(all_frames, output_path, fps, music_path, total_sec)
    else:
        _write_video_only(all_frames, output_path, fps)

    console.print(f"\n[bold green]Done! Final reel: {output_path}[/bold green]")


def _write_video_only(frames: list[np.ndarray], out_path: str, fps: int):
    import imageio
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=9, macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()


def _write_with_audio(frames: list[np.ndarray], out_path: str, fps: int, music_path: str, duration: float):
    """
    Write video frames then mux in background audio using FFmpeg subprocess.
    Audio is looped/trimmed to match video duration and mixed at -18 dBFS.
    """
    import subprocess

    tmp_video = out_path.replace(".mp4", "_noaudio.mp4")
    _write_video_only(frames, tmp_video, fps)

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-stream_loop", "-1",
        "-i", music_path,
        "-t", str(duration),
        "-filter_complex", "[1:a]volume=-18dB,afade=t=out:st={}:d=2[a]".format(max(duration - 2, 0)),
        "-map", "0:v",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    log.info(f"FFmpeg mux: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"FFmpeg error:\n{result.stderr}")
        console.print("[yellow]Audio mux failed; keeping video-only output.[/yellow]")
        import shutil
        shutil.move(tmp_video, out_path)
    else:
        os.remove(tmp_video)


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--config", default="config.yaml", help="Path to config.yaml")
@click.option("--transition", default=8, help="Crossfade length in frames (0 = hard cut)")
@click.option("--music", default=None, help="Optional background audio file path")
@click.option("--output", default=None, help="Override output filename")
def main(config: str, transition: int, music: Optional[str], output: Optional[str]):
    """Assemble scene clips into the final MekongTunnel reel."""
    cfg = load_config(config)
    scenes = load_scenes(cfg["scenes_file"])

    out_dir = cfg["output"]["dir"]
    final_name = output or cfg["output"]["final_filename"]
    final_path = str(Path(out_dir) / final_name)
    fps = cfg["output"]["fps"]

    # Discover clips in scene order
    scene_clips = []
    missing = []
    for s in scenes:
        mp4_path = str(Path(out_dir) / f"{s['id']}.mp4")
        scene_clips.append((s["id"], mp4_path))
        if not Path(mp4_path).exists():
            missing.append(mp4_path)

    if missing:
        console.print(f"[yellow]Warning: {len(missing)} clip(s) not found:[/yellow]")
        for m in missing:
            console.print(f"  {m}")
        console.print("Run generate.py first, or they will be skipped.")

    assemble(scene_clips, final_path, fps, transition_frames=transition, music_path=music)


if __name__ == "__main__":
    main()
