#!/usr/bin/env python3
"""Generate a Twitter-ready animation of one Baby-vLLM engine step."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1200, 675
FPS = 12
DURATION = 12.0

# Everything is drawn at SCALE and downsampled, which is what gives the
# rounded corners and small type clean edges in the final GIF.
SCALE = 2

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "assets" / "twitter" / "baby_vllm_engine_step.gif"
PREVIEW = ROOT / "assets" / "twitter" / "baby_vllm_preview.png"

BG = "#FCFDFE"
CARD = "#FFFFFF"
SOFT = "#F2F6FA"
LINE = "#DBE3EC"
SHADOW = "#EDF1F6"
INK = "#26313D"
MUTED = "#5F7183"
ACCENT = "#2F6FA8"

AMBER = "#EDAE63"
BLUE = "#7BB0DC"
VIOLET = "#A990DC"
GREEN = "#63AC7F"
FREE = "#EBF0F5"

AMBER_INK = "#96601A"
BLUE_INK = "#2A6494"
VIOLET_INK = "#5B4795"
GREEN_INK = "#2F7A50"

FONT_REGULAR = "/System/Library/Fonts/SFNS.ttf"
FONT_ROUNDED = "/System/Library/Fonts/SFNSRounded.ttf"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, *, rounded: bool = False, mono: bool = False):
    path = FONT_MONO if mono else FONT_ROUNDED if rounded else FONT_REGULAR
    return ImageFont.truetype(path, size * SCALE)


F11 = font(11)
F12 = font(12)
F13 = font(13)
F14 = font(14)
F15 = font(15)
F17 = font(17)
F19 = font(19, rounded=True)
F30 = font(30, rounded=True)
M11 = font(11, mono=True)
M12 = font(12, mono=True)
M13 = font(13, mono=True)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease(value: float) -> float:
    """Ease-out-cubic: motion arrives softly instead of stopping dead."""
    value = clamp(value)
    return 1 - (1 - value) ** 3


def phase(t: float, start: float, end: float) -> float:
    return ease((t - start) / (end - start))


def lerp(a: float, b: float, amount: float) -> float:
    return a + (b - a) * clamp(amount)


def color_mix(a: str, b: str, amount: float) -> str:
    a = a.lstrip("#")
    b = b.lstrip("#")
    rgb = [
        round(lerp(int(a[i : i + 2], 16), int(b[i : i + 2], 16), amount)) for i in (0, 2, 4)
    ]
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def s(value: float) -> float:
    return value * SCALE


def sbox(box) -> tuple[float, ...]:
    return tuple(value * SCALE for value in box)


def card(draw, box, radius=14, fill=CARD, outline=LINE, width=1, shadow=True):
    if shadow:
        x1, y1, x2, y2 = box
        draw.rounded_rectangle(
            sbox((x1, y1 + 2, x2, y2 + 3)), radius=s(radius), fill=SHADOW
        )
    draw.rounded_rectangle(
        sbox(box), radius=s(radius), fill=fill, outline=outline, width=round(s(width))
    )


def text(draw, xy, value, *, f=F14, fill=INK, anchor="la"):
    draw.text((s(xy[0]), s(xy[1])), value, font=f, fill=fill, anchor=anchor)


def tracked_text(draw, xy, value, *, f=F11, fill=MUTED, tracking=1.6):
    """Uppercase labels read better with a little letter spacing."""
    x, y = s(xy[0]), s(xy[1])
    for char in value:
        draw.text((x, y), char, font=f, fill=fill, anchor="lm")
        x += f.getlength(char) + s(tracking)


def arrow(draw, start, end, color=MUTED, width=1.5, head=6):
    x1, y1 = s(start[0]), s(start[1])
    x2, y2 = s(end[0]), s(end[1])
    draw.line((x1, y1, x2, y2), fill=color, width=round(s(width)))
    angle = math.atan2(y2 - y1, x2 - x1)
    for offset in (2.5, -2.5):
        draw.line(
            (
                x2,
                y2,
                x2 + s(head) * math.cos(angle + offset),
                y2 + s(head) * math.sin(angle + offset),
            ),
            fill=color,
            width=round(s(width)),
        )


def chip(draw, box, label, fill, ink=INK, f=F13, outline=None):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(
        sbox(box),
        radius=s((y2 - y1) / 2),
        fill=fill,
        outline=outline or color_mix(fill, INK, 0.14),
        width=round(s(1)),
    )
    text(draw, ((x1 + x2) / 2, (y1 + y2) / 2), label, f=f, fill=ink, anchor="mm")


def token_box(draw, x, y, w, h, label, fill, ink=INK, f=F12):
    draw.rounded_rectangle(
        sbox((x, y, x + w, y + h)),
        radius=s(6),
        fill=fill,
        outline=color_mix(fill, INK, 0.18),
        width=round(s(1)),
    )
    text(draw, (x + w / 2, y + h / 2), label, f=f, fill=ink, anchor="mm")


def dot_column(draw, x, y, fill, outline, filled, total=4, active=0.0):
    """The shared visual language for a sequence: a column of token slots."""
    draw.rounded_rectangle(
        sbox((x, y, x + 40, y + 128)), radius=s(11), fill=fill, outline=outline, width=round(s(1))
    )
    for node in range(total):
        cy = y + 19 + node * 30
        r = 5.5 if node in filled else 4.5
        node_fill = active if node in filled else color_mix(CARD, LINE, 0.75)
        draw.ellipse(
            sbox((x + 20 - r, cy - r, x + 20 + r, cy + r)),
            fill=node_fill,
            outline=color_mix(node_fill, INK, 0.12),
            width=round(s(1)),
        )


# ---------------------------------------------------------------- timeline
T_QUEUE = 0.0
T_BATCH = 2.0
T_EXEC = 3.8
T_CACHE = 5.2
T_SAMPLE = 7.4
T_REUSE = 9.9
T_CLEAR = 11.2


def reset_amount(t: float) -> float:
    """The panels wind back at the very end so the GIF loops without a jump."""
    return phase(t, T_CLEAR + 0.1, DURATION - 0.05)


def draw_header(draw):
    text(draw, (48, 36), "Baby-vLLM", f=F30, fill=INK, anchor="lm")
    text(draw, (214, 38), "one engine step", f=F15, fill=MUTED, anchor="lm")
    text(
        draw,
        (1152, 38),
        "github.com/mushyalpha/baby-vllm",
        f=M12,
        fill=MUTED,
        anchor="rm",
    )
    draw.line(sbox((48, 66, 1152, 66)), fill=LINE, width=round(s(1)))


PANELS = [
    (48, 248, "1", "REQUESTS"),
    (274, 510, "2", "SCHEDULER"),
    (536, 816, "3", "MODEL EXECUTOR"),
    (842, 1152, "4", "SAMPLED OUTPUT"),
]


def draw_panels(draw, t):
    stages = [T_QUEUE, T_BATCH, T_EXEC, T_SAMPLE]
    for i, (x1, x2, number, label) in enumerate(PANELS):
        lit = t >= stages[i]
        badge = ACCENT if lit else color_mix(LINE, MUTED, 0.25)
        draw.ellipse(sbox((x1, 86, x1 + 18, 104)), fill=badge)
        text(draw, (x1 + 9, 95), number, f=M11, fill=CARD, anchor="mm")
        tracked_text(draw, (x1 + 26, 96), label, f=F11, fill=MUTED)
        card(draw, (x1, 112, x2, 348))
    arrow(draw, (254, 230), (268, 230), LINE, 1.5, 6)
    arrow(draw, (516, 230), (530, 230), LINE, 1.5, 6)
    arrow(draw, (822, 230), (836, 230), LINE, 1.5, 6)


def draw_requests(draw, t):
    # The columns are always on screen so the loop has no pop; arrival order
    # is told with a highlight that sweeps A, then B, then C.
    done = phase(t, T_REUSE + 0.4, T_REUSE + 1.2) * (1 - reset_amount(t))
    sequences = [
        ("A", AMBER, AMBER_INK, {3}, "decode", 0.2),
        ("B", BLUE, BLUE_INK, {0, 1, 2, 3}, "prefill", 0.7),
        ("C", VIOLET, VIOLET_INK, {1, 2, 3}, "waiting", 1.2),
    ]
    for i, (name, color, ink, filled, kind, start) in enumerate(sequences):
        sweep = phase(t, start, start + 0.5) * (1 - phase(t, start + 0.9, start + 1.5))
        retired = done if name == "A" else 0.0
        x = 72 + i * 56
        body = color_mix(color_mix(SOFT, color, 0.16 + 0.16 * sweep), SOFT, retired)
        edge = color_mix(color_mix(LINE, color, 0.28 + 0.42 * sweep), LINE, retired)
        node = color_mix(color_mix(color, FREE, 0.35 * (1 - sweep)), FREE, retired)
        dot_column(draw, x, 136, body, edge, filled, active=node)
        label_ink = color_mix(color_mix(MUTED, ink, sweep), MUTED, retired)
        text(draw, (x + 20, 280), name, f=F14, fill=label_ink, anchor="mm")
        state = "done" if retired > 0.5 else kind
        text(draw, (x + 20, 297), state, f=F11, fill=MUTED, anchor="mm")
    text(draw, (148, 328), "arrival order is kept", f=F11, fill=MUTED, anchor="mm")


def draw_scheduler(draw, t):
    # The budget empties again at the end, which both fills the tail of the
    # loop with motion and returns the panel to its opening state.
    clear = phase(t, T_CLEAR, T_CLEAR + 0.7)
    fill_up = phase(t, T_BATCH, T_BATCH + 1.0) * (1 - clear)
    card(draw, (294, 136, 490, 206), 11, SOFT, LINE, 1, shadow=False)
    tracked_text(draw, (312, 154), "TOKEN BUDGET", f=F11, fill=MUTED)
    used = round(5 * fill_up)
    for i in range(8):
        slot = (310 + i * 21, 170, 326 + i * 21, 190)
        colour = FREE
        if i < used:
            colour = AMBER if i == 0 else BLUE
        draw.rounded_rectangle(
            sbox(slot), radius=s(4), fill=colour, outline=color_mix(colour, INK, 0.12), width=round(s(1))
        )
    text(draw, (392, 232), "continuous batch", f=F17, fill=INK, anchor="mm")
    if fill_up > 0.02:
        amber = color_mix(CARD, AMBER, fill_up)
        blue = color_mix(CARD, BLUE, fill_up)
        chip(draw, (306, 254, 372, 284), "A · 1", amber, color_mix(CARD, INK, fill_up))
        text(draw, (386, 269), "+", f=F15, fill=color_mix(CARD, MUTED, fill_up), anchor="mm")
        chip(draw, (400, 254, 478, 284), "B · 4", blue, color_mix(CARD, INK, fill_up))
    budget = "5 of 8 tokens this step" if fill_up > 0.5 else "waiting for the next step"
    text(draw, (392, 328), budget, f=F11, fill=MUTED, anchor="mm")


def draw_executor(draw, t):
    run = phase(t, T_EXEC, T_EXEC + 1.2)
    # Stay lit through the sampling beat; the pass is what produced the logits.
    fade = 1 - phase(t, T_REUSE, T_REUSE + 0.8)
    for i in range(4):
        x = 575 + i * 54
        local = clamp(run * 1.35 - i * 0.14) * fade
        body = color_mix(SOFT, BLUE, 0.10 + 0.34 * local)
        edge = color_mix(LINE, BLUE_INK, 0.30 * local)
        node = color_mix(color_mix(FREE, BLUE_INK, 0.30), BLUE_INK, local)
        dot_column(draw, x, 140, body, edge, {0, 1, 2, 3}, active=node)
        if i < 3:
            arrow(
                draw,
                (x + 42, 204),
                (x + 52, 204),
                color_mix(LINE, MUTED, 0.35 + 0.5 * local),
                1.5,
                5,
            )
    text(draw, (683, 292), "forward pass", f=F14, fill=INK, anchor="mm")
    text(draw, (683, 328), "one flat token batch", f=F11, fill=MUTED, anchor="mm")

    travel = phase(t, T_EXEC - 0.8, T_EXEC + 0.3)
    if 0 < travel < 1:
        for index, (label, color) in enumerate(
            [("A", AMBER), ("B", BLUE), ("B", BLUE), ("B", BLUE), ("B", BLUE)]
        ):
            local = clamp(travel * 1.3 - index * 0.06)
            x = lerp(496, 578 + (index % 4) * 15, local)
            y = lerp(216 + (index - 2) * 7, 196, local)
            token_box(draw, x, y, 21, 21, label, color, f=F11)


LOGITS = [0.34, 0.52, 0.27, 0.88, 0.42, 0.61, 0.31, 0.46]
ARGMAX = 3


def draw_output(draw, t):
    # The whole panel is scaffolded from frame one (ghost bars and empty token
    # slots) so it never reads as an unfinished, empty box.
    settle = 1 - reset_amount(t)
    grow = phase(t, T_SAMPLE, T_SAMPLE + 0.9) * settle
    picked = phase(t, T_SAMPLE + 0.9, T_SAMPLE + 1.3) * settle
    tracked_text(draw, (866, 138), "LOGITS OVER VOCABULARY", f=F11, fill=MUTED)
    baseline = 210
    for i, height in enumerate(LOGITS):
        x = 866 + i * 19
        # Before the pass the bars sit low and near-flat, not as empty stubs.
        top = baseline - lerp(height * 16, height * 56, grow)
        is_max = i == ARGMAX
        base = color_mix(FREE, BLUE, 0.55 * grow)
        colour = color_mix(base, AMBER, picked if is_max else 0)
        draw.rounded_rectangle(
            sbox((x, top, x + 14, baseline)),
            radius=s(3),
            fill=colour,
            outline=color_mix(colour, INK, 0.12),
            width=round(s(1)),
        )
    draw.line(sbox((864, baseline, 1015, baseline)), fill=LINE, width=round(s(1)))
    x_max = 866 + ARGMAX * 19 + 7
    text(draw, (x_max, 224), "argmax", f=F11, fill=color_mix(CARD, AMBER_INK, picked), anchor="mm")

    reveal = phase(t, T_SAMPLE + 1.2, T_SAMPLE + 1.8) * settle
    if reveal > 0.02:
        arrow(draw, (1022, 182), (1036, 182), color_mix(CARD, LINE, reveal), 1.5, 6)
        token_box(
            draw,
            lerp(1030, 1040, reveal),
            167,
            74,
            30,
            "cache",
            color_mix(CARD, BLUE, reveal),
            color_mix(CARD, INK, reveal),
            f=F13,
        )

    draw.line(sbox((866, 242, 1128, 242)), fill=LINE, width=round(s(1)))
    rows = [
        ("A", '"!"', AMBER, AMBER_INK, "done", GREEN_INK, T_SAMPLE + 1.5),
        ("B", '"cache"', BLUE, BLUE_INK, "running", BLUE_INK, T_SAMPLE + 1.8),
    ]
    for i, (name, tok, color, ink, status, status_ink, start) in enumerate(rows):
        appear = phase(t, start, start + 0.5) * settle
        y = 254 + i * 34
        text(draw, (866, y + 13), name, f=F13, fill=ink, anchor="lm")
        draw.rounded_rectangle(
            sbox((888, y, 958, y + 26)), radius=s(6), fill=FREE, outline=LINE, width=round(s(1))
        )
        if appear > 0.02:
            token_box(
                draw,
                888,
                y,
                70,
                26,
                tok,
                color_mix(FREE, color, appear),
                color_mix(FREE, INK, appear),
                f=F12,
            )
            text(draw, (972, y + 13), status, f=F11, fill=color_mix(CARD, status_ink, appear), anchor="lm")
    text(draw, (997, 330), "sample, append, repeat", f=F11, fill=MUTED, anchor="mm")


BLOCK_OWNER = {
    1: ("B", BLUE, BLUE_INK),
    2: ("A", AMBER, AMBER_INK),
    5: ("B", BLUE, BLUE_INK),
    7: ("A", AMBER, AMBER_INK),
    9: ("B", BLUE, BLUE_INK),
}


def draw_kv_cache(draw, t):
    card(draw, (48, 372, 1152, 618))
    text(draw, (72, 400), "Paged KV cache", f=F19, fill=INK, anchor="lm")
    chip(draw, (222, 387, 330, 413), "block size = 4", SOFT, MUTED, M11, LINE)
    tracked_text(draw, (1026, 400), "PHYSICAL BLOCKS", f=F11, fill=MUTED)

    allocated = phase(t, T_CACHE - 0.2, T_CACHE + 1.0) * (1 - reset_amount(t))
    writes = phase(t, T_CACHE + 0.6, T_CACHE + 2.1)
    freed = phase(t, T_REUSE + 0.2, T_REUSE + 1.3)

    positions = {}
    for i in range(12):
        row, col = divmod(i, 6)
        x = 416 + col * 118
        y = 428 + row * 72
        positions[i] = (x, y)
        owner, color, owner_ink = BLOCK_OWNER.get(i, ("", FREE, MUTED))
        amount = allocated if i in BLOCK_OWNER else 0.0
        if i in (2, 7):
            amount *= 1 - freed
        body = color_mix(FREE, color, 0.42 * amount)
        edge = color_mix(LINE, color, 0.7 * amount)
        draw.rounded_rectangle(
            sbox((x, y, x + 104, y + 56)), radius=s(9), fill=body, outline=edge, width=round(s(1))
        )
        text(draw, (x + 11, y + 13), f"#{i:02}", f=M11, fill=MUTED, anchor="lm")
        if amount > 0.35:
            text(draw, (x + 93, y + 13), owner, f=F12, fill=owner_ink, anchor="rm")
            for slot in range(4):
                filled = clamp(writes * 1.6 - (i * 0.03 + slot * 0.07))
                slot_fill = color_mix(CARD, color, 0.25 + 0.7 * filled)
                draw.rounded_rectangle(
                    sbox((x + 11 + slot * 21, y + 30, x + 27 + slot * 21, y + 44)),
                    radius=s(3),
                    fill=slot_fill,
                    outline=color_mix(slot_fill, INK, 0.12),
                    width=round(s(1)),
                )
        else:
            text(draw, (x + 52, y + 36), "free", f=F11, fill=MUTED, anchor="mm")

    tracked_text(draw, (72, 434), "BLOCK TABLES", f=F11, fill=MUTED)
    tables = [("A", [2, 7], AMBER, AMBER_INK), ("B", [1, 5, 9], BLUE, BLUE_INK)]
    for row, (label, ids, color, ink) in enumerate(tables):
        y = 470 + row * 46
        released = freed if label == "A" else 0.0
        text(draw, (72, y), label, f=F14, fill=color_mix(ink, MUTED, released), anchor="lm")
        arrow(draw, (90, y), (110, y), LINE, 1.5, 5)
        for col, block_id in enumerate(ids):
            body = color_mix(FREE, color, 0.42 * allocated * (1 - released))
            chip(
                draw,
                (120 + col * 44, y - 13, 154 + col * 44, y + 13),
                str(block_id),
                body,
                MUTED if released > 0.5 else INK,
                M12,
            )

    if 0 < writes < 1:
        for index, (block_id, color) in enumerate(
            [(2, AMBER), (7, AMBER), (1, BLUE), (5, BLUE), (9, BLUE)]
        ):
            bx, by = positions[block_id]
            local = clamp(writes * 1.55 - index * 0.09)
            if local <= 0 or local >= 1:
                continue
            x = lerp(676, bx + 34, local)
            y = lerp(376, by + 17, local)
            token_box(draw, x, y, 36, 22, "K/V", color, f=F11)

    text(draw, (72, 570), "One sequence, many blocks.", f=F12, fill=MUTED, anchor="lm")
    text(draw, (72, 590), "Memory need not be contiguous.", f=F13, fill=INK, anchor="lm")


CAPTIONS = [
    (T_QUEUE, T_BATCH, "Three requests are queued.", "Arrival order decides priority."),
    (T_BATCH, T_EXEC, "The scheduler packs decode", "and prefill into one batch."),
    (T_EXEC, T_CACHE, "One forward pass runs", "the whole flat batch."),
    (T_CACHE, T_SAMPLE, "Fresh K/V lands in", "reusable cache blocks."),
    (T_SAMPLE, T_REUSE, "Argmax picks one token", "for every sequence."),
    (T_REUSE, DURATION, "Finished work frees blocks.", "The next step reuses them."),
]


def draw_caption(draw, t):
    card(draw, (416, 560, 1128, 600), 10, SOFT, LINE, 1, shadow=False)
    current = CAPTIONS[-1]
    for item in CAPTIONS:
        if item[0] <= t < item[1]:
            current = item
            break
    start, end, line1, line2 = current
    fade = min(phase(t, start, start + 0.3), 1 - phase(t, end - 0.3, end))
    draw.rounded_rectangle(sbox((416, 560, 420, 600)), radius=s(2), fill=color_mix(SOFT, ACCENT, fade))
    text(draw, (438, 572), line1, f=F13, fill=color_mix(SOFT, INK, fade), anchor="lm")
    text(draw, (438, 589), line2, f=F12, fill=color_mix(SOFT, MUTED, fade), anchor="lm")


STAGES = [
    ("queue", T_QUEUE),
    ("batch", T_BATCH),
    ("execute", T_EXEC),
    ("cache", T_CACHE),
    ("sample", T_SAMPLE),
    ("reuse", T_REUSE),
]


def draw_progress(draw, t):
    if reset_amount(t) > 0.5:
        t = 0.0
    x = 48
    for i, (label, start) in enumerate(STAGES):
        end = STAGES[i + 1][1] if i + 1 < len(STAGES) else DURATION
        active = start <= t < end
        passed = t >= end
        fill = ACCENT if active else (color_mix(LINE, ACCENT, 0.45) if passed else LINE)
        r = 4 if active else 3
        draw.ellipse(sbox((x, 644 - r, x + 2 * r, 644 + r)), fill=fill)
        label_x = x + 2 * r + 7
        text(
            draw,
            (label_x, 645),
            label,
            f=F11,
            fill=INK if active else MUTED,
            anchor="lm",
        )
        x = label_x + F11.getlength(label) / SCALE + 22


def draw_frame(t: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw)
    draw_panels(draw, t)
    draw_requests(draw, t)
    draw_scheduler(draw, t)
    draw_executor(draw, t)
    draw_output(draw, t)
    draw_kv_cache(draw, t)
    draw_caption(draw, t)
    draw_progress(draw, t)
    return image.resize((WIDTH, HEIGHT), Image.LANCZOS)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [draw_frame(index / FPS) for index in range(round(DURATION * FPS))]
    preview_index = min(round(FPS * (T_SAMPLE + 2.0)), len(frames) - 1)
    frames[preview_index].save(PREVIEW, optimize=True)

    # One shared palette across frames keeps the flat fills from shifting.
    palette_source = frames[preview_index].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    palette = palette_source.getpalette()
    quantized = []
    for frame in frames:
        paletted = frame.quantize(palette=palette_source, dither=Image.Dither.NONE)
        paletted.putpalette(palette)
        quantized.append(paletted)

    quantized[0].save(
        OUTPUT,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"Wrote {PREVIEW}")


if __name__ == "__main__":
    main()
