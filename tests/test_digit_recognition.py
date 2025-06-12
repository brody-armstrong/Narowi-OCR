import cv2
import numpy as np
import os
import pytest
from src.processing.image_processor import ImageProcessor
from src.processing.ocr_engine import OCREngine
from src.processing.validator import Validator, ValidationIssue


def generate_synthetic_image(text: str, output_path: str, width: int = 100, height: int = 50):
    """
    Generates a synthetic image with the given text.
    Args:
        text: The text to draw on the image.
        output_path: Path to save the generated PNG image.
        width: Width of the image.
        height: Height of the image.
    """
    # Create a white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Choose a common sans-serif font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0)  # Black

    # Get text size to center it
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    # Draw the text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the image
    cv2.imwrite(output_path, image)


def test_3_5_recognition_full_preprocess_v1():
    """
    Tests OCR performance on synthetic images of digits '3' and '5'
    using the full v1 preprocessing pipeline (CLAHE, Bilateral Filter, Unsharp Masking, Morph Ops).
    """
    image_dir = "tests/test_data/digit_confusion/"
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])

    img_processor = ImageProcessor()
    ocr_engine = OCREngine()
    default_psm_mode = ocr_engine.get_psm()

    print("\nOCR Results for 3-5 Recognition after Morphological Operations (with PSM adjustments):")
    all_tests_passed = True
    for image_path in image_files:
        filename = os.path.basename(image_path)
        expected_text = filename.replace("img_", "").replace(".png", "")
        current_psm_for_ocr = default_psm_mode

        if filename in ["img_3.png", "img_5.png"]:
            ocr_engine.set_psm('10') # PSM 10: Treat the image as a single character.
            current_psm_for_ocr = '10'
        else:
            ocr_engine.set_psm(default_psm_mode)

        # Load image
        image = img_processor.load_image(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            all_tests_passed = False
            continue

        # Preprocess image
        preprocessed_image = img_processor.preprocess_for_ocr(image)
        if preprocessed_image is None:
            print(f"Failed to preprocess image: {image_path}")
            all_tests_passed = False
            continue

        # Extract text and confidence
        extracted_text, confidence = ocr_engine.extract_with_confidence(preprocessed_image)

        print(f"File: {filename}, PSM: {current_psm_for_ocr}, Extracted: '{extracted_text}', Expected: '{expected_text}', Confidence: {confidence:.2f}")

        if filename == "img_3.png":
            try:
                assert extracted_text == "3", f"For {filename}, expected '3', got '{extracted_text}'"
                assert confidence > 50, f"For {filename}, confidence {confidence} not > 50"
            except AssertionError as e:
                print(f"AssertionError: {e}")
                all_tests_passed = False
        elif filename == "img_5.png":
            try:
                assert extracted_text == "5", f"For {filename}, expected '5', got '{extracted_text}'"
                assert confidence > 50, f"For {filename}, confidence {confidence} not > 50"
            except AssertionError as e:
                print(f"AssertionError: {e}")
                all_tests_passed = False
        else:
            try:
                assert extracted_text == expected_text, f"For {filename}, expected '{expected_text}', got '{extracted_text}'"
                assert confidence > 80, f"For {filename}, confidence {confidence} not > 80"
            except AssertionError as e:
                print(f"AssertionError: {e}")
                all_tests_passed = False

    assert all_tests_passed, "One or more assertions failed. Check output above."


def test_char_confidence_extraction_and_validation(capsys): # capsys still useful for other prints if needed
    img_path = os.path.join("tests", "test_data", "digit_confusion", "img_835.png")
    assert os.path.exists(img_path), f"Test image not found: {img_path}"

    processor = ImageProcessor()
    engine = OCREngine()
    default_psm = engine.get_psm()
    engine.set_psm(default_psm)

    image = processor.load_image(img_path)
    assert image is not None, "Failed to load test image"

    preprocessed_image = processor.preprocess_for_ocr(image)

    word_data = engine.extract_detailed_data(preprocessed_image)
    assert word_data is not None and len(word_data) > 0

    # Assuming '835' is recognized as a single word, which it was previously.
    # This is important for how Validator.LOW_CONFIDENCE_THRESHOLD_3_5 is tested.
    assert len(word_data) == 1, "Expected '835' to be a single word in word_data"
    word_info_835 = word_data[0]
    assert word_info_835['text'] == '835', f"Expected word '835', got '{word_info_835['text']}'"

    # Print word info for debugging if needed
    print(f"Word: '{word_info_835['text']}', Conf: {word_info_835['conf']:.2f}")

    reconstructed_text_from_data = "".join([wi['text'] for wi in word_data])
    text_from_extract_text, _ = engine.extract_with_confidence(preprocessed_image)
    text_from_extract_text = text_from_extract_text.strip()

    assert reconstructed_text_from_data == "835"
    assert text_from_extract_text == "835"

    # Test Validator's low-confidence issue creation
    original_threshold = Validator.LOW_CONFIDENCE_THRESHOLD_3_5
    Validator.LOW_CONFIDENCE_THRESHOLD_3_5 = 95.0 # '835' usually has ~94% conf

    # Pass word_data (which is a list of dicts)
    validated_text, issues = Validator.validate_digits_with_confidence(word_data, text_from_extract_text)
    assert validated_text == text_from_extract_text

    assert len(issues) == 2, f"Expected 2 issues, got {len(issues)}. Issues: {issues}"

    issue_for_3_found = False
    issue_for_5_found = False
    for issue in issues:
        assert issue.context_type == 'confidence_check'
        assert issue.word_text == '835'
        # Confidence in message should match word_info_835['conf']
        # Using f-string formatting for float comparison in string might be tricky, check substring
        assert f"Low confidence for digit '{issue.original_char}'" in issue.message

        if issue.original_char == '3':
            assert issue.char_index_in_word == 1 # '3' is at index 1 in "835"
            issue_for_3_found = True
        elif issue.original_char == '5':
            assert issue.char_index_in_word == 2 # '5' is at index 2 in "835"
            issue_for_5_found = True

    assert issue_for_3_found, "ValidationIssue for digit '3' not found."
    assert issue_for_5_found, "ValidationIssue for digit '5' not found."

    Validator.LOW_CONFIDENCE_THRESHOLD_3_5 = original_threshold # Reset


def test_medical_range_validation():
    validator = Validator() # Instance not strictly needed for classmethod, but fine

    # Test case 1: High temperature
    high_temp_data = [{'text': '153.2', 'conf': 96.0, 'level': 5, 'page_num': 1, 'block_num': 1, 'par_num': 1, 'line_num': 1, 'word_num': 1, 'left': 0, 'top': 0, 'width': 10, 'height': 10}]
    _, high_temp_issues = validator.validate_digits_with_confidence(high_temp_data, "153.2", context="temperature_fahrenheit")
    assert len(high_temp_issues) == 1
    assert "Potential out-of-range temperature: 153.2F" in high_temp_issues[0].message
    assert high_temp_issues[0].context_type == 'range_check'
    assert high_temp_issues[0].word_text == '153.2'

    # Test case 2: Normal temperature (should have no range issue)
    normal_temp_data = [{'text': '98.6', 'conf': 97.0,  'level': 5, 'page_num': 1, 'block_num': 1, 'par_num': 1, 'line_num': 1, 'word_num': 1, 'left': 0, 'top': 0, 'width': 10, 'height': 10}]
    _, normal_temp_issues = validator.validate_digits_with_confidence(normal_temp_data, "98.6", context="temperature_fahrenheit")
    range_issues = [issue for issue in normal_temp_issues if issue.context_type == 'range_check']
    assert len(range_issues) == 0

    # Test case 3: Low temperature
    low_temp_data = [{'text': '85.0', 'conf': 94.0,  'level': 5, 'page_num': 1, 'block_num': 1, 'par_num': 1, 'line_num': 1, 'word_num': 1, 'left': 0, 'top': 0, 'width': 10, 'height': 10}]
    original_threshold = Validator.LOW_CONFIDENCE_THRESHOLD_3_5
    Validator.LOW_CONFIDENCE_THRESHOLD_3_5 = 80.0 # Ensure no confidence issue for '5'
    _, low_temp_issues = validator.validate_digits_with_confidence(low_temp_data, "85.0", context="temperature_fahrenheit")
    Validator.LOW_CONFIDENCE_THRESHOLD_3_5 = original_threshold

    # Filter for range_check issues specifically, as there might be confidence issues if not careful with threshold
    low_temp_range_issues = [issue for issue in low_temp_issues if issue.context_type == 'range_check']
    assert len(low_temp_range_issues) == 1
    assert "Potential out-of-range temperature: 85.0F" in low_temp_range_issues[0].message
    assert low_temp_range_issues[0].context_type == 'range_check'
    assert low_temp_range_issues[0].word_text == '85.0'

    # Test case 4: Non-numeric text with temperature context (should not error out)
    text_temp_data = [{'text': 'Hot', 'conf': 90.0,  'level': 5, 'page_num': 1, 'block_num': 1, 'par_num': 1, 'line_num': 1, 'word_num': 1, 'left': 0, 'top': 0, 'width': 10, 'height': 10}]
    _, text_issues = validator.validate_digits_with_confidence(text_temp_data, "Hot", context="temperature_fahrenheit")
    text_range_issues = [issue for issue in text_issues if issue.context_type == 'range_check']
    assert len(text_range_issues) == 0