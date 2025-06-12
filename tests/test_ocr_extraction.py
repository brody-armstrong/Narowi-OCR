import os
import pytest
from src.processing.image_processor import ImageProcessor
from src.processing.ocr_engine import OCREngine

def test_ocr_extraction():
    """Test OCR extraction from a medical device image."""
    # Get the directory where this test file is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the test image relative to the test file
    image_path = os.path.join(test_dir, 'thermometer_synth_1.png')

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