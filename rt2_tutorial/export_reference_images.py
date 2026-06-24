from __future__ import annotations

import struct
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

from rt2_tokenizer import RT2Tokenizer
from synthetic_data import MixedRT2Dataset


FONT_5X7: dict[str, list[str]] = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
    ",": ["00000", "00000", "00000", "00000", "00100", "00100", "01000"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "01110", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "<": ["00010", "00100", "01000", "10000", "01000", "00100", "00010"],
    ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
    "[": ["01110", "01000", "01000", "01000", "01000", "01000", "01110"],
    "]": ["01110", "00010", "00010", "00010", "00010", "00010", "01110"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "a": ["00000", "01110", "00001", "01111", "10001", "10011", "01101"],
    "b": ["10000", "10000", "10110", "11001", "10001", "10001", "11110"],
    "c": ["00000", "01110", "10001", "10000", "10000", "10001", "01110"],
    "d": ["00001", "00001", "01101", "10011", "10001", "10001", "01111"],
    "e": ["00000", "01110", "10001", "11111", "10000", "10001", "01110"],
    "f": ["00110", "01001", "01000", "11100", "01000", "01000", "01000"],
    "g": ["00000", "01111", "10001", "10001", "01111", "00001", "01110"],
    "h": ["10000", "10000", "10110", "11001", "10001", "10001", "10001"],
    "i": ["00100", "00000", "01100", "00100", "00100", "00100", "01110"],
    "j": ["00010", "00000", "00110", "00010", "00010", "10010", "01100"],
    "k": ["10000", "10010", "10100", "11000", "10100", "10010", "10001"],
    "l": ["01100", "00100", "00100", "00100", "00100", "00100", "01110"],
    "m": ["00000", "11010", "10101", "10101", "10101", "10101", "10101"],
    "n": ["00000", "10110", "11001", "10001", "10001", "10001", "10001"],
    "o": ["00000", "01110", "10001", "10001", "10001", "10001", "01110"],
    "p": ["00000", "11110", "10001", "10001", "11110", "10000", "10000"],
    "q": ["00000", "01101", "10011", "10001", "01111", "00001", "00001"],
    "r": ["00000", "10110", "11001", "10000", "10000", "10000", "10000"],
    "s": ["00000", "01111", "10000", "01110", "00001", "00001", "11110"],
    "t": ["01000", "01000", "11100", "01000", "01000", "01001", "00110"],
    "u": ["00000", "10001", "10001", "10001", "10001", "10011", "01101"],
    "v": ["00000", "10001", "10001", "10001", "10001", "01010", "00100"],
    "w": ["00000", "10001", "10001", "10101", "10101", "10101", "01010"],
    "x": ["00000", "10001", "01010", "00100", "01010", "10001", "10001"],
    "y": ["00000", "10001", "10001", "01111", "00001", "00001", "01110"],
    "z": ["00000", "11111", "00010", "00100", "01000", "10000", "11111"],
}


def wrap_text(text: str, max_chars: int) -> list[str]:
    """
    按大致字符数做简单换行。
    """

    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = word if not current else f"{current} {word}"
        if len(proposal) <= max_chars:
            current = proposal
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_char(
    canvas: torch.Tensor,
    x: int,
    y: int,
    char: str,
    color: tuple[float, float, float],
    scale: int = 1,
) -> None:
    """
    用 5x7 点阵字形在画布上绘制单个字符。
    """

    pattern = FONT_5X7.get(char.lower(), FONT_5X7[" "])
    height = canvas.size(1)
    width = canvas.size(2)
    for row_idx, row in enumerate(pattern):
        for col_idx, bit in enumerate(row):
            if bit != "1":
                continue
            for dy in range(scale):
                for dx in range(scale):
                    px = x + col_idx * scale + dx
                    py = y + row_idx * scale + dy
                    if 0 <= px < width and 0 <= py < height:
                        canvas[0, py, px] = color[0]
                        canvas[1, py, px] = color[1]
                        canvas[2, py, px] = color[2]


def draw_text(
    canvas: torch.Tensor,
    x: int,
    y: int,
    text: str,
    color: tuple[float, float, float],
    scale: int = 1,
) -> None:
    """
    在画布上画一行文本。
    """

    cursor_x = x
    for char in text:
        draw_char(canvas, cursor_x, y, char, color, scale=scale)
        cursor_x += 6 * scale


def resize_nearest(image: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    把输入图像按最近邻放大到目标尺寸。

    用最近邻是因为我们的合成图本身像素风格很强，
    放大后仍然会保持干净的块状边界，适合教学展示。
    """

    image = image.unsqueeze(0)
    resized = F.interpolate(image, size=(target_size, target_size), mode="nearest")
    return resized.squeeze(0)


def save_rgb_png(path: Path, image: torch.Tensor) -> None:
    """
    用标准库把 RGB Tensor 存成 PNG。

    输入:
      image: [3, H, W], 数值范围 [0, 1]
    """

    image_u8 = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    _, height, width = image_u8.shape

    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)
        row = image_u8[:, y, :].permute(1, 0).contiguous().view(-1).tolist()
        raw_rows.extend(row)

    compressed = zlib.compress(bytes(raw_rows), level=9)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(
        chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
    )
    png.extend(chunk(b"IDAT", compressed))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def compose_reference_image(image: torch.Tensor, title: str, lines: list[str]) -> torch.Tensor:
    """
    在原图下方追加说明区域，并把文字直接画上去。
    """

    image = resize_nearest(image, target_size=512)
    _, height, width = image.shape

    font_scale = 3
    text_line_height = 10 * font_scale
    top_padding = 16
    bottom_padding = 16
    panel_height = top_padding + bottom_padding + text_line_height * (1 + len(lines))
    canvas = torch.ones(3, height + panel_height, width)
    canvas[:, :height, :] = image

    # 标题用蓝色，正文用黑色，便于快速区分。
    draw_text(canvas, 12, height + top_padding, title, (0.1, 0.2, 0.8), scale=font_scale)
    for idx, line in enumerate(lines):
        draw_text(
            canvas,
            12,
            height + top_padding + text_line_height * (idx + 1),
            line,
            (0.0, 0.0, 0.0),
            scale=font_scale,
        )
    return canvas


def main() -> None:
    """
    导出若干带注释的参考图片，供学习 RT-2 教学代码使用。
    """

    tokenizer = RT2Tokenizer(num_action_dims=8, action_bins=64)
    dataset = MixedRT2Dataset(
        tokenizer=tokenizer,
        num_samples=16,
        use_chain_of_thought=False,
    )

    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "rt2_reference_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = [0, 1, 2, 3, 5, 7]
    for export_idx, dataset_idx in enumerate(sample_indices, start=1):
        sample = dataset[dataset_idx]
        image = sample["images"][0].clone()  # type: ignore[index]
        prompt_text = str(sample["prompt_text"])
        task_type = str(sample["task_type"])
        # 512 像素宽画布下，3 倍放大的 5x7 点阵字大致每字符占 18 像素。
        # 留出左右边距后，这里控制每行 26 个字符左右，基本能完整显示。
        max_chars = 26

        if task_type == "vqa":
            target_text = tokenizer.decode_text(sample["target_ids"].tolist(), skip_special_tokens=False)  # type: ignore[union-attr]
            title = f"[sample {export_idx}] vqa"
            details = [
                *wrap_text(f"prompt: {prompt_text}", max_chars=max_chars),
                *wrap_text(f"target: {target_text}", max_chars=max_chars),
            ]
        else:
            action_values = tokenizer.decode_action_token_ids(sample["target_ids"].tolist())  # type: ignore[union-attr]
            action_text = ", ".join(f"{float(x):.2f}" for x in action_values.tolist())
            raw_token_text = tokenizer.decode_text(sample["target_ids"].tolist(), skip_special_tokens=False)  # type: ignore[union-attr]
            title = f"[sample {export_idx}] robot"
            details = [
                *wrap_text(f"prompt: {prompt_text}", max_chars=max_chars),
                *wrap_text(f"tokens: {raw_token_text}", max_chars=max_chars),
                *wrap_text(f"action: {action_text}", max_chars=max_chars),
            ]

        composed = compose_reference_image(image, title, details)
        save_path = output_dir / f"sample_{export_idx:02d}_{task_type}.png"
        save_rgb_png(save_path, composed)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
