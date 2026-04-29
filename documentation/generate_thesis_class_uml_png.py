from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


OUTPUT_PATH = Path(__file__).with_name("thesis_class_uml.png")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    weight = "bold" if bold else "normal"
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight))
    return ImageFont.truetype(path, size=size)


TITLE_FONT = _font(42, bold=True)
HEADER_FONT = _font(28, bold=True)
BODY_FONT = _font(20, bold=False)
SMALL_FONT = _font(18, bold=False)


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.fill(line, width=width) for line in text.splitlines())


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, spacing: int = 6) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_box(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: str,
    outline: str,
) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=24, fill=fill, outline=outline, width=4)

    title_text = _wrap(title, 22)
    body_text = _wrap(body, 28)
    title_w, title_h = _measure(draw, title_text, HEADER_FONT)
    body_w, body_h = _measure(draw, body_text, BODY_FONT)

    title_x = x1 + (x2 - x1 - title_w) // 2
    title_y = y1 + 18
    body_x = x1 + (x2 - x1 - body_w) // 2
    body_y = title_y + title_h + 16

    draw.multiline_text((title_x, title_y), title_text, font=HEADER_FONT, fill="#102030", align="center", spacing=6)
    draw.multiline_text((body_x, body_y), body_text, font=BODY_FONT, fill="#102030", align="center", spacing=6)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, dash: bool = False) -> None:
    draw.line([start, end], fill="#314559", width=4, joint="curve")
    if dash:
        # Keep the line visually light without introducing extra dependencies.
        pass
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux = dx / length
    uy = dy / length
    arrow_size = 16
    left = (end[0] - int(arrow_size * ux) + int(arrow_size * uy / 2), end[1] - int(arrow_size * uy) - int(arrow_size * ux / 2))
    right = (end[0] - int(arrow_size * ux) - int(arrow_size * uy / 2), end[1] - int(arrow_size * uy) + int(arrow_size * ux / 2))
    draw.polygon([end, left, right], fill="#314559")


def main() -> None:
    width, height = 2400, 1600
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, width - 1, height - 1), outline="#d8dde6", width=2)
    draw.text((70, 40), "Credit Risk Prediction System", font=TITLE_FONT, fill="#102030")
    subtitle = "Class UML overview derived from the implemented Python codebase"
    draw.text((70, 98), subtitle, font=SMALL_FONT, fill="#4d5d70")

    # Top config layer
    _draw_box(draw, xy=(90, 190, 610, 360), title="Config", body="SECRET_KEY\nDEBUG\nHOST\nPORT", fill="#e8f2ff", outline="#7aa6d9")
    _draw_box(draw, xy=(680, 190, 1200, 360), title="ModelConfig", body="TEST_SIZE\nRANDOM_STATE\nCV_FOLDS\nHyperparameter maps", fill="#edf7ec", outline="#7ab97a")
    _draw_box(draw, xy=(1270, 190, 1790, 360), title="ExplainabilityConfig", body="SHAP_SAMPLES\nLIME_SAMPLES\nTOP_FEATURES", fill="#fff3e0", outline="#e0a75f")

    # Web / orchestration layer
    _draw_box(draw, xy=(820, 430, 1580, 620), title="CreditRiskPredictor", body="predict()\nget_instance()\navailable_models()\ninput_schema()\nuses ExplainabilityConfig", fill="#f2eefc", outline="#8d74d1")
    _draw_box(draw, xy=(1800, 430, 2290, 620), title="web.app", body="Flask app\nuses Config\n/schema /models\n/predict /health", fill="#eef8ff", outline="#68a5d6")

    # Pipeline layer
    _draw_box(draw, xy=(80, 790, 560, 980), title="input_validator", body="validate_input()\nget_input_schema()", fill="#f9f4ff", outline="#ac84dd")
    _draw_box(draw, xy=(650, 790, 1150, 980), title="preprocessor", body="build_preprocessor()\nprepare_data()\nload_preprocessor()", fill="#ecfff6", outline="#62b58a")
    _draw_box(draw, xy=(1240, 790, 1740, 980), title="train_models", body="train_all_models()\nload_model()\nensure_trained_models()", fill="#fff8ea", outline="#d1a84d")
    _draw_box(draw, xy=(1830, 790, 2340, 980), title="shap_explainer", body="create_explainer()\nexplain_prediction()\nget_global_importance()", fill="#ffeef0", outline="#d27b87")

    # Lower supporting layer
    _draw_box(draw, xy=(80, 1110, 560, 1290), title="data_loader", body="load_german_credit()\nget_feature_target_split()", fill="#f0fbff", outline="#5ca6bd")
    _draw_box(draw, xy=(650, 1110, 1150, 1290), title="evaluate", body="evaluate_model()\ncross_validate_model()\ncomparison_table()", fill="#f6f6f6", outline="#8a8a8a")
    _draw_box(draw, xy=(1240, 1110, 1740, 1290), title="LimeExplainerWrapper", body="explain_prediction()\nplot_explanation()", fill="#fff0e7", outline="#cf8a5d")
    _draw_box(draw, xy=(1830, 1110, 2340, 1290), title="lime_explainer", body="create_lime_explainer()", fill="#fff0e7", outline="#cf8a5d")

    # Connectors based on the actual code relationships.
    _arrow(draw, (980, 620), (980, 790))
    _arrow(draw, (1090, 620), (1090, 790))
    _arrow(draw, (1210, 620), (1460, 790))
    _arrow(draw, (1520, 620), (2080, 790))
    _arrow(draw, (860, 620), (330, 790))
    _arrow(draw, (1130, 980), (330, 1110))
    _arrow(draw, (830, 980), (900, 1110))
    _arrow(draw, (1400, 980), (990, 1110))
    _arrow(draw, (1500, 430), (1520, 360))

    # web app dispatches to the predictor.
    _arrow(draw, (2040, 430), (1290, 430))

    canvas.save(OUTPUT_PATH, format="PNG", dpi=(180, 180))
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()