# vdo-gen — MekongTunnel Product Reel Generator

Local Python pipeline that generates a **25-second vertical 9:16 product demo video**
for MekongTunnel CLI using open-source AI video models and GPU-accelerated inference.

Brand style: warm beige/cream background · gold-orange accent (`#C58A2B`) · clean SaaS minimal.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× RTX 3090 (24 GB) | 3× RTX 4090 (72 GB total) |
| RAM | 32 GB | 64 GB |
| Storage | 40 GB free | 80 GB free |
| OS | Ubuntu 20.04+ / macOS 13+ | Ubuntu 22.04 |
| CUDA | 11.8+ | 12.1+ |

---

## Setup

### 1 — Clone and create environment

```bash
git clone https://github.com/yourname/vdo-gen
cd vdo-gen
python3 -m venv .venv
source .venv/bin/activate
```

### 2 — Install PyTorch with CUDA

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4 — Install FFmpeg (for final encode and audio mux)

```bash
# Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 5 — Install DejaVu fonts (for overlay rendering)

```bash
# Ubuntu
sudo apt install fonts-dejavu-core fonts-dejavu-extra

# macOS — fonts are already present via Homebrew/system
```

---

## Model Downloads

Models are downloaded automatically from Hugging Face on first run.
Pre-download to avoid timeouts during generation:

```bash
# AnimateDiff (default, ~8 GB)
python - <<'EOF'
from diffusers import AnimateDiffPipeline, MotionAdapter
MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3")
AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism")
EOF

# Wan2.1 (alternative, ~5 GB)
python - <<'EOF'
from diffusers import WanPipeline
WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B")
EOF
```

---

## Project Structure

```
vdo-gen/
├── generate.py          # Generate per-scene clips (AI model + overlays)
├── stitch_video.py      # Assemble clips into final reel
├── config.yaml          # Model, output, brand, overlay settings
├── requirements.txt
├── prompts/
│   └── scenes.yaml      # Scene definitions (prompts, timing, overlays)
├── overlays/
│   ├── __init__.py
│   └── renderer.py      # Pillow-based brand overlay renderer
├── output/              # Generated clips and final reel
└── logs/                # Generation logs
```

---

## Configuration

Edit `config.yaml` to change:

| Key | Description |
|-----|-------------|
| `model.name` | `animatediff` \| `wan` \| `svd` |
| `model.gpu_ids` | List of GPU indices, e.g. `[0, 1, 2]` |
| `generation.seed` | Global seed (per-scene offset is added automatically) |
| `brand.*` | All brand colours in hex |
| `output.fps` | Frame rate (default 24) |

---

## Running the Pipeline

### Full pipeline (recommended)

```bash
# Step 1 — Generate all scene clips
python generate.py

# Step 2 — Stitch into final reel
python stitch_video.py
```

Output: `output/mekongtunnel_reel.mp4`

### Dry run (see scene plan without generating)

```bash
python generate.py --dry-run
```

### Generate specific scenes only

```bash
python generate.py --scenes scene_01,scene_04
```

### Sequential mode (single GPU)

```bash
python generate.py --sequential
```

### Custom output filename

```bash
python stitch_video.py --output my_reel.mp4
```

### With background music

```bash
python stitch_video.py --music path/to/music.mp3
```

### Hard cuts instead of crossfades

```bash
python stitch_video.py --transition 0
```

---

## Scene Plan

| Scene | Time | Content | Overlay |
|-------|------|---------|---------|
| scene_01 | 0–3s | Google search for mekongtunnel.dev | Animated typing + result card |
| scene_02 | 3–6s | Website hero section | Brand heading card |
| scene_03 | 6–10s | Terminal: `mekong 3000` | Animated terminal block |
| scene_04 | 10–15s | Terminal output with public URL | Gold-highlighted URL |
| scene_05 | 15–19s | Browser opens public URL | Browser chrome bar |
| scene_06 | 19–22s | `mekong ps` / `mekong deploy` flash | Cycling terminal sequences |
| scene_07 | 22–25s | End card | Fade-in brand typography |

---

## Overlay System

All text overlays are rendered entirely in Python (Pillow) — **never by the AI model**.
This guarantees readable terminal commands, URLs, and brand text at every frame.

Overlay types in `scenes.yaml`:

| Type | Description |
|------|-------------|
| `google_search` | Light Google UI with typing animation and result card |
| `hero_card` | Warm beige brand hero heading with CTA button |
| `terminal_command` | Floating terminal window with animated typing |
| `terminal_output` | Terminal with multi-line output and optional gold URL highlight |
| `terminal_multi` | Rapidly cycles through command sequences |
| `browser_url` | Minimal browser address bar |
| `end_card` | Centered brand typography with fade-in |

---

## Troubleshooting

**CUDA out of memory**
```yaml
# config.yaml
model:
  offload_to_cpu: true
```

**Slow generation on single GPU**
```bash
python generate.py --sequential
```

**Font not found**
```bash
sudo apt install fonts-dejavu-core fonts-dejavu-extra fonts-liberation
```

**FFmpeg not found**
```bash
which ffmpeg  # check it's on PATH
sudo apt install ffmpeg
```

**Model download fails**
```bash
export HF_HUB_OFFLINE=0
huggingface-cli login  # if using gated models
```

---

## Example Full Run

```bash
source .venv/bin/activate

# Preview the scene plan
python generate.py --dry-run

# Generate all clips (parallel, 3× RTX 4090)
python generate.py

# Assemble final reel with 10-frame crossfades
python stitch_video.py --transition 10

# Open result
xdg-open output/mekongtunnel_reel.mp4
```
