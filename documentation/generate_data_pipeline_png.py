from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


OUTPUT_PATH = Path(__file__).with_name("data_pipeline_diagram.png")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    weight = "bold" if bold else "normal"
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight))
    return ImageFont.truetype(path, size=size)


TITLE_FONT = _font(40, bold=True)
HEADER_FONT = _font(24, bold=True)
BODY_FONT = _font(18, bold=False)


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, spacing: int = 6) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _box(draw: ImageDraw.ImageDraw, xy, title: str, body: str, fill: str, outline: str) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=22, fill=fill, outline=outline, width=4)
    title_w, title_h = _measure(draw, title, HEADER_FONT)
    body_w, body_h = _measure(draw, body, BODY_FONT)
    draw.text((x1 + (x2 - x1 - title_w) // 2, y1 + 18), title, font=HEADER_FONT, fill="#102030")
    draw.multiline_text(
        (x1 + (x2 - x1 - body_w) // 2, y1 + 18 + title_h + 14),
        body,
        font=BODY_FONT,
        fill="#102030",
        align="center",
        spacing=6,
    )


def _arrow(draw: ImageDraw.ImageDraw, start, end) -> None:
    draw.line([start, end], fill="#314559", width=4)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux = dx / length
    uy = dy / length
    arrow_size = 15
    left = (end[0] - int(arrow_size * ux) + int(arrow_size * uy / 2), end[1] - int(arrow_size * uy) - int(arrow_size * ux / 2))
    right = (end[0] - int(arrow_size * ux) - int(arrow_size * uy / 2), end[1] - int(arrow_size * uy) + int(arrow_size * ux / 2))
    draw.polygon([end, left, right], fill="#314559")


def main() -> None:
    canvas = Image.new("RGB", (2400, 1200), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 2399, 1199), outline="#d8dde6", width=2)
    draw.text((70, 40), "Data Transformation Pipeline", font=TITLE_FONT, fill="#102030")

    _box(draw, (70, 170, 390, 320), "Raw JSON Input", "20 fields\nApplicant payload", "#eef8ff", "#68a5d6")
    _box(draw, (470, 170, 790, 320), "Input Validator", "Required fields\nRanges & categories", "#f9f4ff", "#ac84dd")
    _box(draw, (870, 170, 1180, 320), "Pandas DataFrame", "Single-row\nvalidated record", "#ecfff6", "#62b58a")

    _box(draw, (70, 470, 510, 650), "Numerical pipeline", "Median imputer\nStandardScaler", "#e8f2ff", "#7aa6d9")
    _box(draw, (580, 470, 1020, 650), "Categorical pipeline", "Constant imputer\nOneHotEncoder", "#fff3e0", "#e0a75f")
    _box(draw, (1090, 470, 1470, 650), "Merged features", "Transformed matrix\nN x 57", "#f2eefc", "#8d74d1")

    _box(draw, (1560, 470, 1900, 650), "ML model", "Logistic Regression\nRandom Forest\nXGBoost", "#fff8ea", "#d1a84d")
    _box(draw, (1990, 470, 2330, 650), "Prediction output", "Class label\nProbability", "#ffeef0", "#d27b87")

    _box(draw, (1560, 810, 2330, 980), "SHAP explanation", "Feature contributions\nTop positive / negative factors", "#f0fbff", "#5ca6bd")

    _arrow(draw, (390, 245), (470, 245))
    _arrow(draw, (790, 245), (870, 245))
    _arrow(draw, (1030, 320), (290, 470))
    _arrow(draw, (1030, 320), (800, 470))
    _arrow(draw, (1180, 245), (1260, 470))
    _arrow(draw, (1470, 560), (1560, 560))
    _arrow(draw, (1900, 560), (1990, 560))
    _arrow(draw, (1740, 650), (1740, 810))

    canvas.save(OUTPUT_PATH, format="PNG", dpi=(180, 180))
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()