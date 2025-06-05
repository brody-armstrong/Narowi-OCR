import cv2
import numpy as np
import os
import random

FONT = cv2.FONT_HERSHEY_SIMPLEX
IMG_WIDTH = 300
IMG_HEIGHT = 120
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

OUTPUT_DIR = "synthetic_images/scale"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEIGHT_UNITS = ["lb", "lbs", "kg", "kgs"]


def generate_weight_image(weight: float, unit: str, filename: str):
    img = np.full((IMG_HEIGHT, IMG_WIDTH, 3), BG_COLOR, dtype=np.uint8)
    text = f"{weight}{unit}"
    font_scale = 2.2 if len(str(int(weight))) < 3 else 1.8
    thickness = 4
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    x = (IMG_WIDTH - text_width) // 2
    y = (IMG_HEIGHT + text_height) // 2
    cv2.putText(img, text, (x, y), FONT, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)


def main():
    # Generate valid weights
    for i in range(30):
        unit = random.choice(WEIGHT_UNITS)
        if "kg" in unit:
            weight = round(random.uniform(20, 250), 1)
        else:
            weight = round(random.uniform(50, 500), 1)
        filename = f"valid_{i}_{weight}{unit}.png"
        generate_weight_image(weight, unit, filename)

    # Generate invalid weights (too low, too high, malformed)
    invalid_cases = [
        ("5lb", "invalid_low_lb.png"),
        ("600lb", "invalid_high_lb.png"),
        ("3kg", "invalid_low_kg.png"),
        ("300kg", "invalid_high_kg.png"),
        ("abc", "invalid_text.png"),
        ("lb", "invalid_novalue.png"),
        ("123.", "invalid_trailingdot.png"),
        (".kg", "invalid_leadingdot.png"),
    ]
    for text, filename in invalid_cases:
        img = np.full((IMG_HEIGHT, IMG_WIDTH, 3), BG_COLOR, dtype=np.uint8)
        font_scale = 2.0
        thickness = 4
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
        x = (IMG_WIDTH - text_width) // 2
        y = (IMG_HEIGHT + text_height) // 2
        cv2.putText(img, text, (x, y), FONT, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)

if __name__ == "__main__":
    main() 