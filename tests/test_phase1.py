import pytest
import numpy as np
import cv2
import os # Added for path manipulation if needed, though create_test_image returns array
from src.processing.image_processor import ImageProcessor
from src.processing.ocr_engine import OCREngine
from src.processing.number_extractor import NumberExtractor, ExtractedNumber # Assuming this is the correct import
from src.processing.validator import Validator, ValidationIssue

def create_test_image():
    """Create a test image with medical readings."""
    # Create a white background (3-channel BGR image)
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255 # Height increased slightly for new line
    
    # Add some text
    cv2.putText(img, "BP: 120/80", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Temp: 98.6F", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "O2: 98%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "ID: 351", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) # New line
    
    return img

def test_image_processor():
    """Test image processing functions."""
    processor = ImageProcessor()
    
    # Create test image
    test_img = create_test_image()
    
    # Test validation
    assert processor.validate_image(test_img) == True
    assert processor.validate_image(None) == False
    
    # Test resize
    resized = processor.resize_image(test_img, max_dimension=300)
    assert resized.shape[0] <= 300 or resized.shape[1] <= 300
    
    # Test preprocessing
    preprocessed = processor.preprocess_for_ocr(test_img)
    # Check that the output is grayscale (2D) and has the same height and width
    assert len(preprocessed.shape) == 2
    assert preprocessed.shape[0] == test_img.shape[0]
    assert preprocessed.shape[1] == test_img.shape[1]
    assert preprocessed.dtype == np.uint8

def test_ocr_engine():
    """Test OCR functionality."""
    engine = OCREngine()
    
    # Create test image
    test_img = create_test_image()
    
    # Test text extraction
    text = engine.extract_text(test_img)
    assert isinstance(text, str)
    
    # Test confidence
    confidence = engine.get_confidence(test_img)
    assert 0 <= confidence <= 100
    
    # Test combined extraction
    text, conf = engine.extract_with_confidence(test_img)
    assert isinstance(text, str)
    assert 0 <= conf <= 100

def test_number_extractor():
    """Test number extraction functionality."""
    extractor = NumberExtractor()
    
    # Test blood pressure extraction
    bp_text = "BP: 120/80"
    numbers = extractor.extract_numbers(bp_text, confidence=90.0)
    assert len(numbers) == 2
    assert numbers[0].value == 120
    assert numbers[1].value == 80
    assert numbers[0].unit == 'mmHg'
    
    # Test temperature extraction
    temp_text = "Temp: 98.6F"
    numbers = extractor.extract_numbers(temp_text, confidence=90.0)
    assert len(numbers) == 1
    assert numbers[0].value == 98.6
    assert numbers[0].unit == '°F'
    
    # Test validation
    valid_bp = ExtractedNumber(value=120, unit='mmHg', confidence=90.0, raw_text="120/80")
    invalid_bp = ExtractedNumber(value=300, unit='mmHg', confidence=90.0, raw_text="300/80")
    assert extractor.validate_reading(valid_bp) == True
    assert extractor.validate_reading(invalid_bp) == False

def test_integration():
    """Test integration of all components."""
    # Create test image
    test_img = create_test_image()
    
    # Process image
    processor = ImageProcessor()
    preprocessed = processor.preprocess_for_ocr(test_img)
    
    # Perform OCR
    engine = OCREngine() # Using default config, PSM might need adjustment for block text
    # Consider engine = OCREngine(config={'--psm': '6'}) if full block OCR is better
    text, confidence = engine.extract_with_confidence(preprocessed)
    print(f"Raw OCR Text:\n{text}")
    print(f"Overall Confidence: {confidence:.2f}%")

    word_data = engine.extract_detailed_data(preprocessed)
    assert word_data is not None, "extract_detailed_data failed"
    # print(f"Word Data: {word_data}") # Can be verbose

    # Validation
    validated_text, issues = Validator.validate_digits_with_confidence(word_data, text)
    print(f"Initial Validation Issues: {[issue.message for issue in issues]}")

    # Test confidence validation on "351"
    id_word_info_list = [wd for wd in word_data if "351" in wd['text']]
    assert id_word_info_list, "Word containing '351' not found in word_data"
    id_word_info = id_word_info_list[0] # Take the first match

    original_threshold = Validator.LOW_CONFIDENCE_THRESHOLD_3_5
    Validator.LOW_CONFIDENCE_THRESHOLD_3_5 = id_word_info['conf'] + 5.0
    
    _, id_issues = Validator.validate_digits_with_confidence(word_data, text)

    confidence_issues_for_351 = [
        issue for issue in id_issues
        if issue.word_text == id_word_info['text'] and issue.context_type == 'confidence_check'
    ]

    found_3_issue = any(issue.original_char == '3' for issue in confidence_issues_for_351)
    found_5_issue = any(issue.original_char == '5' for issue in confidence_issues_for_351)

    # Depending on how "ID: 351" is OCR'd (e.g. "ID:", "351" or "ID:351")
    # We expect issues for '3' and '5' within the word that contains "351".
    assert found_3_issue, f"Confidence issue for '3' in '{id_word_info['text']}' not found. Issues: {confidence_issues_for_351}"
    assert found_5_issue, f"Confidence issue for '5' in '{id_word_info['text']}' not found. Issues: {confidence_issues_for_351}"
    
    Validator.LOW_CONFIDENCE_THRESHOLD_3_5 = original_threshold

    # Test temperature range validation
    temp_word_data_list = []
    temp_word_text_for_validation = ""
    for wd in word_data:
        # Flexible check for temperature, could be "98.6F", "98.6" etc.
        if "98.6" in wd['text'] or ("Temp:" in wd['text'] and len(wd['text']) > 5):
            temp_word_data_list = [{k:v for k,v in wd.items()}]
            # Attempt to isolate the numeric part for validation context
            # This is a simplification; real scenarios might need more robust parsing here
            temp_word_text_for_validation = wd['text'].replace('F','').replace('Temp:','').strip()
            break

    assert temp_word_data_list, "Temperature word '98.6' not found in word_data"

    _, temp_issues = Validator.validate_digits_with_confidence(temp_word_data_list, temp_word_text_for_validation, context="temperature_fahrenheit")
    range_check_issues_normal_temp = [issue for issue in temp_issues if issue.context_type == 'range_check']
    assert not range_check_issues_normal_temp, f"Expected no range issues for normal temp '{temp_word_text_for_validation}', got {range_check_issues_normal_temp}"

    fake_temp_word_data = [{'text': '150', 'conf': 95.0, 'level': 5, 'page_num': 1, 'block_num': 1, 'par_num': 1, 'line_num': 1, 'word_num': 1, 'left': 0, 'top': 0, 'width': 10, 'height': 10}]
    _, fake_temp_issues = Validator.validate_digits_with_confidence(fake_temp_word_data, "150", context="temperature_fahrenheit")
    range_check_issues_fake_temp = [issue for issue in fake_temp_issues if issue.context_type == 'range_check']
    assert len(range_check_issues_fake_temp) == 1, f"Expected 1 range issue for fake temp '150', got {len(range_check_issues_fake_temp)}"
    assert "Potential out-of-range temperature: 150F" in range_check_issues_fake_temp[0].message

    # Extract numbers (original NumberExtractor assertions)
    extractor = NumberExtractor()
    numbers = extractor.extract_numbers(text, confidence) # text is the full OCR'd block
    print(f"Extracted Numbers (NumberExtractor): {[(n.value, n.unit) for n in numbers]}")

    # Verify results (original NumberExtractor assertions)
    assert len(numbers) > 0 # Should find BP, Temp, O2, ID

    # Check that specific numbers were extracted by NumberExtractor
    # This part needs to align with how NumberExtractor works and what it returns.
    # The original test had `assert extractor.validate_reading(number)`.
    # Let's check if expected values are present.
    extracted_values_types = set()
    for num_obj in numbers:
        if num_obj.value == 120 and num_obj.unit == "mmHg": extracted_values_types.add("BP_sys")
        if num_obj.value == 80 and num_obj.unit == "mmHg": extracted_values_types.add("BP_dia")
        if num_obj.value == 98.6 and num_obj.unit == "°F": extracted_values_types.add("TempF")
        if num_obj.value == 98 and num_obj.unit == "%": extracted_values_types.add("O2")
        # For "ID: 351", NumberExtractor might extract 351. Assuming it has no unit or a generic one.
        if num_obj.value == 351: extracted_values_types.add("ID")


    assert "BP_sys" in extracted_values_types, "Systolic BP not extracted by NumberExtractor"
    assert "BP_dia" in extracted_values_types, "Diastolic BP not extracted by NumberExtractor"
    assert "TempF" in extracted_values_types, "Temperature in F not extracted by NumberExtractor"
    assert "O2" in extracted_values_types, "O2 saturation not extracted by NumberExtractor"
    assert "ID" in extracted_values_types, "ID '351' not extracted by NumberExtractor"

    for number in numbers:
        assert isinstance(number, ExtractedNumber)
        # The original test_integration had this line:
        # assert extractor.validate_reading(number)
        # This implies NumberExtractor has its own validation logic. We keep it.
        if not extractor.validate_reading(number):
            print(f"NumberExtractor validation failed for: {number.value}{number.unit} (Raw: {number.raw_text})")
        assert extractor.validate_reading(number) 