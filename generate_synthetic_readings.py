import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_reading_image(text, font_path, font_size, img_size=(400, 200), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """Generates an image with the given text."""
    img = Image.new('RGB', img_size, color=bg_color)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to a more robust font path for macOS or default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
            print("Using system Arial font.")
        except IOError:
            font = ImageFont.load_default()
            print(f"Warning: Could not load font. Using default font.")

    bbox = d.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (img_size[0] - text_width) / 2
    y = (img_size[1] - text_height) / 2

    d.text((x, y), text, fill=text_color, font=font)
    return np.array(img)

def main():
    output_dir = 'synthetic_images/generated_readings'
    os.makedirs(output_dir, exist_ok=True)

    # Use a default font or specify a path to a .ttf font
    font_path = "arial.ttf" # This path is now less critical as system path is tried first
    font_size = 60

    readings_to_generate = [
        "98.6F", "37.0C", "101.5F",
        "150lb", "70kg", "WT: 65.5kg", "180 lbs",
        "120/80 mmHg", "BP 130/90",
        "72bpm", "HR: 85bpm", "90 BPM"
    ]

    for i, reading in enumerate(readings_to_generate):
        img_array = generate_reading_image(reading, font_path, font_size)
        cv2.imwrite(os.path.join(output_dir, f"reading_{i}_{reading.replace('/', '_').replace(':', '')}.png"), img_array)
    print(f"Generated {len(readings_to_generate)} synthetic reading images in {output_dir}")

if __name__ == "__main__":
    main() 