import os
import pytest
import cv2
from src.processing.ocr_engine import OCREngine
from src.processing.pattern_matcher import PatternMatcher

@pytest.fixture
def ocr_engine():
    return OCREngine(device_type='scale')

@pytest.fixture
def pattern_matcher():
    return PatternMatcher()

def test_ocr_on_synthetic_scale_images(ocr_engine, pattern_matcher):
    # Directory containing synthetic scale images
    image_dir = "synthetic_images/scale"
    assert os.path.exists(image_dir), f"Directory {image_dir} does not exist"

    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if not filename.endswith('.png'):
            continue
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        assert img is not None, f"Failed to load image {image_path}"

        # Extract text using OCR
        text, confidence = ocr_engine.extract_with_confidence(img)
        assert text, f"OCR failed to extract text from {filename}"

        # Extract weight readings from the OCR text
        readings = pattern_matcher.extract_weight(text, confidence)
        assert readings, f"No weight readings extracted from {filename}"

        # For valid images, verify the extracted value and unit
        if filename.startswith('valid_'):
            reading = readings[0]
            assert reading.is_valid, f"Invalid reading extracted from {filename}"
            print(f"Image: {filename}, Extracted: {reading.value} {reading.unit}, Confidence: {confidence}")

        # For invalid images, verify that the reading is marked as invalid
        elif filename.startswith('invalid_'):
            for reading in readings:
                assert not reading.is_valid, f"Valid reading extracted from {filename}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 