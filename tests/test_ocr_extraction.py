import os
import pytest
from src.processing.image_processor import ImageProcessor
from src.processing.ocr_engine import OCREngine

def test_ocr_extraction():
    """Test OCR extraction from a medical device image."""
    # Path to the test image
    image_path = '/Users/brody/Downloads/Narowi/medical_ocr/tests/thermometer.png'  # Replace with the actual path to your test image

    # Check if the image file exists
    assert os.path.exists(image_path), f"Test image not found at {image_path}"

    # Initialize the image processor and OCR engine
    processor = ImageProcessor()
    engine = OCREngine()

    # Load and preprocess the image
    image = processor.load_image(image_path)
    preprocessed = processor.preprocess_for_ocr(image)

    # Extract text from the preprocessed image
    text, confidence = engine.extract_with_confidence(preprocessed)

    # Assert that text was extracted
    assert text, "No text was extracted from the image."

    # Print the extracted text and confidence
    print(f"Extracted Text: {text}")
    print(f"Confidence: {confidence}")

    # Optionally, assert that the confidence is above a certain threshold
    assert confidence > 0.5, "OCR confidence is too low." 