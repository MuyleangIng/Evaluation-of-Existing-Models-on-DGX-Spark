"""
overlays/renderer.py
────────────────────
Brand-aware overlay renderer for the MekongTunnel product reel.

Brand: warm beige/cream backgrounds, gold-orange accent (#C58A2B),
       clean minimal SaaS aesthetic. No neon. No cyberpunk.

All text is drawn with Pillow — never delegated to the AI model —
so every word is pixel-perfect and readable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# ─── Font discovery ───────────────────────────────────────────────────────────

_SANS_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]

_SANS_LIGHT_PATHS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-L.ttf",
]

_MONO_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]


def _load_font(paths: list[str], size: int) -> ImageFont.FreeTypeFont:
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def sans(size: int) -> ImageFont.FreeTypeFont:
    return _load_font(_SANS_PATHS, size)

def sans_light(size: int) -> ImageFont.FreeTypeFont:
    return _load_font(_SANS_LIGHT_PATHS, size)

def mono(size: int) -> ImageFont.FreeTypeFont:
    return _load_font(_MONO_PATHS, size)


# ─── Colour helpers ───────────────────────────────────────────────────────────

def hex_rgba(h: str) -> tuple[int, int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = int(h[6:8], 16) if len(h) == 8 else 255
    return r, g, b, a

def hex_rgb(h: str) -> tuple[int, int, int]:
    return hex_rgba(h)[:3]

def with_alpha(h: str, a: int) -> tuple[int, int, int, int]:
    r, g, b, _ = hex_rgba(h)
    return r, g, b, a


# ─── Brand colour shortcuts (lazy-resolved from cfg) ─────────────────────────

def _b(cfg: dict, key: str) -> tuple[int, int, int, int]:
    return hex_rgba(cfg["brand"][key])

def _ba(cfg: dict, key: str, alpha: int) -> tuple[int, int, int, int]:
    return with_alpha(cfg["brand"][key], alpha)


# ─── Warm beige gradient background ─────────────────────────────────────────

def warm_gradient_bg(w: int, h: int, cfg: dict) -> Image.Image:
    """Create a vertical warm beige cream gradient background."""
    top = _b(cfg, "bg_gradient_start")
    bot = _b(cfg, "bg_gradient_end")
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        t = y / h
        arr[y, :] = [
            int(top[0] + (bot[0] - top[0]) * t),
            int(top[1] + (bot[1] - top[1]) * t),
            int(top[2] + (bot[2] - top[2]) * t),
            255,
        ]
    return Image.fromarray(arr, "RGBA")


# ─── Shared helpers ───────────────────────────────────────────────────────────

def centered_x(draw: ImageDraw.ImageDraw, text: str, font, w: int) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return (w - (bbox[2] - bbox[0])) // 2


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def gold_glow(img: Image.Image, draw_fn, glow_color, blur_radius=18) -> Image.Image:
    """Draw text/shape on a separate layer, blur it as a glow, then composite."""
    glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(glow_layer))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(blur_radius))
    return Image.alpha_composite(img, glow_layer)


# ─── Individual overlay drawers ───────────────────────────────────────────────

def draw_caption(draw: ImageDraw.ImageDraw, text: str, w: int, h: int, cfg: dict):
    if not text:
        return
    fnt = sans(cfg["overlay"]["font_size"])
    pad = 32
    tw, th = text_size(draw, text, fnt)
    x = (w - tw) // 2
    y = h - th - pad * 2

    bg = hex_rgba(cfg["overlay"]["caption_bg_color"])
    fg = hex_rgba(cfg["overlay"]["caption_color"])

    draw.rounded_rectangle([x - 24, y - 14, x + tw + 24, y + th + 14],
                            radius=16, fill=bg)
    draw.text((x, y), text, font=fnt, fill=fg)


def draw_google_search(
    draw: ImageDraw.ImageDraw, ov: dict, w: int, h: int, progress: float, cfg: dict
):
    """Google-style light search screen with typing animation and result card."""
    query = ov.get("query", "")
    result_title = ov.get("result_title", "")
    result_url = ov.get("result_url", "")
    animate = ov.get("animate_typing", False)
    click = ov.get("click_animation", False)

    display_q = query[: max(1, int(len(query) * min(progress * 2.5, 1.0)))] if animate else query

    # ── White card panel ────────────────────────────────────────────────────
    card_x1, card_x2 = 32, w - 32
    card_y = h // 5
    draw.rounded_rectangle([card_x1, card_y - 20, card_x2, h - 120],
                            radius=20, fill=(255, 255, 255, 240))

    # ── Google wordmark (coloured letters) ──────────────────────────────────
    g_font = sans(60)
    letters = [("G", "#4285F4"), ("o", "#EA4335"), ("o", "#FBBC05"),
               ("g", "#4285F4"), ("l", "#34A853"), ("e", "#EA4335")]
    gx = w // 2 - 90
    for ch, color in letters:
        draw.text((gx, card_y + 10), ch, font=g_font, fill=hex_rgba(color))
        gx += int(draw.textlength(ch, font=g_font)) + 2

    # ── Search bar ──────────────────────────────────────────────────────────
    bar_y = card_y + 90
    draw.rounded_rectangle([card_x1 + 20, bar_y, card_x2 - 20, bar_y + 72],
                            radius=36, fill=(255, 255, 255, 255),
                            outline=(218, 220, 224, 255), width=2)

    qfont = sans_light(38)
    draw.text((card_x1 + 68, bar_y + 18), display_q, font=qfont, fill=(32, 33, 36, 255))

    # Cursor
    if animate and progress < 0.85:
        cx = card_x1 + 68 + int(draw.textlength(display_q, font=qfont)) + 3
        draw.rectangle([cx, bar_y + 18, cx + 2, bar_y + 52], fill=(32, 33, 36, 255))

    # ── Result card ─────────────────────────────────────────────────────────
    if result_title and progress > 0.55:
        rc_alpha = int(min((progress - 0.55) / 0.35, 1.0) * 255)
        rc_y = bar_y + 100
        # Result URL in green
        url_font = sans_light(30)
        draw.text((card_x1 + 30, rc_y), result_url, font=url_font,
                  fill=(24, 128, 56, rc_alpha))
        # Result title in blue
        rt_font = sans(38)
        draw.text((card_x1 + 30, rc_y + 36), result_title, font=rt_font,
                  fill=(26, 115, 232, rc_alpha))

        # Click ripple
        if click and progress > 0.82:
            ripple_r = int((progress - 0.82) / 0.18 * 50)
            cx_r = card_x1 + 30 + 10
            cy_r = rc_y + 36 + 19
            draw.ellipse([cx_r - ripple_r, cy_r - ripple_r,
                          cx_r + ripple_r, cy_r + ripple_r],
                         outline=(26, 115, 232, max(0, 200 - ripple_r * 4)), width=2)


def draw_hero_card(draw: ImageDraw.ImageDraw, ov: dict, w: int, h: int, progress: float, cfg: dict):
    """Website hero section card with brand style."""
    heading = ov.get("heading", "")
    subheading = ov.get("subheading", "")

    # Soft zoom effect: scale content slightly based on progress
    scale = 1.0 + progress * 0.04

    # Decorative gold accent bar
    bar_w = int(w * 0.12 * scale)
    bar_h = 8
    bx = (w - bar_w) // 2
    by = h // 3 - 60
    draw.rounded_rectangle([bx, by, bx + bar_w, by + bar_h],
                            radius=4, fill=_b(cfg, "accent_gold"))

    # Main heading
    hfont = sans(int(68 * scale))
    htw, hth = text_size(draw, heading, hfont)
    hx = (w - htw) // 2
    hy = h // 3 - 20

    # Shadow
    draw.text((hx + 2, hy + 2), heading, font=hfont,
              fill=_ba(cfg, "text_primary", 60))
    draw.text((hx, hy), heading, font=hfont, fill=_b(cfg, "text_primary"))

    # Subheading in gold accent
    sfont = sans(int(52 * scale))
    stw, _ = text_size(draw, subheading, sfont)
    sx = (w - stw) // 2
    sy = hy + hth + 24
    draw.text((sx, sy), subheading, font=sfont, fill=_b(cfg, "accent_gold"))

    # Decorative pill button
    btn_y = sy + 80
    btn_text = "Get started free →"
    bfont = sans(36)
    btw, bth = text_size(draw, btn_text, bfont)
    bx1 = (w - btw - 60) // 2
    bx2 = bx1 + btw + 60
    draw.rounded_rectangle([bx1, btn_y, bx2, btn_y + bth + 28],
                            radius=28, fill=_b(cfg, "accent_gold"))
    draw.text((bx1 + 30, btn_y + 14), btn_text, font=bfont,
              fill=(255, 252, 245, 255))


def draw_terminal_block(
    draw: ImageDraw.ImageDraw, lines: list[str], w: int, h: int,
    progress: float, animate: bool, cfg: dict,
    highlight_line: int = -1,
):
    """Warm-charcoal terminal window with gold monospace text."""
    fnt = mono(cfg["overlay"]["terminal_font_size"])
    lh = cfg["overlay"]["terminal_font_size"] + 12
    pad = 44
    box_h = len(lines) * lh + pad * 2 + 48
    box_y = h // 2 - box_h // 2
    bx = 36

    bg = _b(cfg, "terminal_bg")
    fg = _b(cfg, "terminal_text")
    hi = _b(cfg, "terminal_highlight")

    # Window shadow
    draw.rounded_rectangle([bx + 6, box_y + 6, w - bx + 6, box_y + box_h + 6],
                            radius=14, fill=(0, 0, 0, 60))
    # Window body
    draw.rounded_rectangle([bx, box_y, w - bx, box_y + box_h],
                            radius=14, fill=bg)
    # Title bar
    draw.rounded_rectangle([bx, box_y, w - bx, box_y + 46],
                            radius=14, fill=with_alpha(cfg["brand"]["terminal_bg"], 255))

    # Traffic light dots
    for ci, col in enumerate([(220, 80, 60, 255), (220, 160, 40, 255), (50, 180, 80, 255)]):
        cx = bx + 20 + ci * 26
        draw.ellipse([cx, box_y + 14, cx + 16, box_y + 30], fill=col)

    # Lines with optional typing animation
    total_chars = sum(len(l) for l in lines)
    shown = int(total_chars * min(progress * 3.0, 1.0)) if animate else total_chars
    char_count = 0

    for li, line in enumerate(lines):
        ty = box_y + 48 + pad // 2 + li * lh
        if animate and char_count >= shown:
            break
        remaining = (shown - char_count) if animate else len(line)
        display_line = line[:remaining]

        color = hi if li == highlight_line else fg
        draw.text((bx + 24, ty), display_line, font=fnt, fill=color)
        char_count += len(line) + 1


def draw_terminal_multi(
    draw: ImageDraw.ImageDraw, ov: dict, w: int, h: int,
    frame_index: int, cfg: dict,
):
    """Rapidly cycle through terminal command sequences."""
    seqs = ov.get("sequences", [])
    if not seqs:
        return

    total_frames = sum(s.get("hold_frames", 24) for s in seqs)
    pos = frame_index % max(total_frames, 1)
    acc = 0
    active_lines = seqs[0].get("lines", [])
    for seq in seqs:
        acc += seq.get("hold_frames", 24)
        if pos < acc:
            active_lines = seq.get("lines", [])
            break

    draw_terminal_block(draw, active_lines, w, h, progress=1.0,
                        animate=False, cfg=cfg)


def draw_browser_bar(draw: ImageDraw.ImageDraw, url: str, w: int, h: int, cfg: dict):
    """Minimal warm browser chrome."""
    bar_h = 74
    draw.rectangle([0, 0, w, bar_h], fill=(240, 235, 225, 255))
    # Tab
    draw.rectangle([0, 46, w, bar_h], fill=(232, 226, 215, 255))
    # Address bar
    draw.rounded_rectangle([72, 8, w - 72, 42], radius=20,
                            fill=(255, 252, 245, 255),
                            outline=(210, 200, 185, 255), width=1)
    fnt = mono(28)
    lock = "🔒 " if url.startswith("https") else ""
    draw.text((90, 12), lock + url, font=fnt, fill=_b(cfg, "text_secondary"))


def draw_end_card(
    draw: ImageDraw.ImageDraw, lines: list[str], w: int, h: int,
    progress: float, cfg: dict
):
    """Warm beige end screen with brand typography and gold accent."""
    alpha = int(min(progress * 2.0, 1.0) * 255)

    # Gold accent divider
    if alpha > 60:
        div_w = int(w * 0.18)
        dv_x = (w - div_w) // 2
        dv_y = h // 2 - 10
        draw.rounded_rectangle([dv_x, dv_y, dv_x + div_w, dv_y + 6],
                                radius=3, fill=(*hex_rgb(cfg["brand"]["accent_gold"]), alpha))

    sizes   = [76, 46, 34]
    colors  = [cfg["brand"]["text_primary"],
               cfg["brand"]["accent_gold"],
               cfg["brand"]["text_light"]]
    gaps    = [100, 72, 0]

    total_h = sum(s + gaps[i] for i, s in enumerate(sizes[:len(lines)]))
    y = (h - total_h) // 2 + 30

    for i, line in enumerate(lines):
        sz  = sizes[i] if i < len(sizes) else 36
        col = colors[i] if i < len(colors) else cfg["brand"]["text_secondary"]
        gap = gaps[i]   if i < len(gaps)   else 60
        fnt = sans(sz)

        tw, th = text_size(draw, line, fnt)
        x = (w - tw) // 2

        # Subtle shadow
        draw.text((x + 2, y + 2), line, font=fnt,
                  fill=(*hex_rgb(cfg["brand"]["bg_secondary"]), alpha // 2))
        draw.text((x, y), line, font=fnt,
                  fill=(*hex_rgb(col), alpha))
        y += sz + gap


# ─── Public entry point ───────────────────────────────────────────────────────

def render_scene_overlay(
    frame: Image.Image,
    scene: dict[str, Any],
    cfg: dict[str, Any],
    frame_index: int,
    total_frames: int,
) -> Image.Image:
    """
    Composite all overlays for one video frame.

    For scenes with a fully-synthetic background (end_card, terminal,
    hero_card), the AI frame is dimmed/replaced with a brand gradient so
    the overlay reads cleanly.
    """
    w, h = cfg["output"]["width"], cfg["output"]["height"]
    progress = frame_index / max(total_frames - 1, 1)

    ov = scene.get("overlay", {})
    ov_type = ov.get("type", "none")

    # Start from AI frame, or swap background for brand scenes
    if ov_type in ("end_card", "hero_card"):
        # Use brand gradient as base (AI frame underneath at low opacity)
        base = warm_gradient_bg(w, h, cfg)
        ai_layer = frame.convert("RGBA")
        ai_layer.putalpha(30)  # very subtle texture
        base = Image.alpha_composite(base, ai_layer)
    else:
        base = frame.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # ── Dispatch overlay type ────────────────────────────────────────────────
    if ov_type == "google_search":
        draw_google_search(draw, ov, w, h, progress, cfg)

    elif ov_type == "hero_card":
        draw_hero_card(draw, ov, w, h, progress, cfg)

    elif ov_type in ("terminal_command", "terminal_output"):
        highlight = ov.get("highlight_line", -1)
        draw_terminal_block(draw, ov.get("lines", []), w, h, progress,
                            animate=ov.get("animate_typing", False),
                            cfg=cfg, highlight_line=highlight)

    elif ov_type == "terminal_multi":
        draw_terminal_multi(draw, ov, w, h, frame_index, cfg)

    elif ov_type == "browser_url":
        draw_browser_bar(draw, ov.get("url", ""), w, h, cfg)

    elif ov_type == "end_card":
        draw_end_card(draw, ov.get("lines", []), w, h, progress, cfg)

    # ── Caption bar (every scene) ────────────────────────────────────────────
    caption = scene.get("caption", "")
    if caption:
        draw_caption(draw, caption, w, h, cfg)

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")
