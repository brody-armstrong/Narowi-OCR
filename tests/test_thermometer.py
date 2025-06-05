import pytest
from src.processing.pattern_matcher import PatternMatcher, ReadingType, MedicalReading

@pytest.fixture
def pattern_matcher():
    return PatternMatcher()

def test_valid_temperature_formats(pattern_matcher):
    """Test various valid temperature formats."""
    valid_cases = [
        ("98.6F", 95.0),
        ("98.6°F", 95.0),
        ("37.0C", 95.0),
        ("37.0°C", 95.0),
        ("102.4F", 95.0),
        ("39.1C", 95.0),
    ]
    
    for text, confidence in valid_cases:
        validation = pattern_matcher.validate_temperature_format(text)
        assert validation.is_valid, f"Failed to validate {text}"
        assert validation.confidence_adjustment == 0.0
        assert validation.error_reason is None
        assert validation.suggested_correction is None

def test_invalid_temperature_formats(pattern_matcher):
    """Test various invalid temperature formats."""
    invalid_cases = [
        ("56./", -20.0, "Invalid character '/' or '\\' in temperature", "56.."),
        ("98.6", -25.0, "Missing temperature unit (F or C)", "98.6°F"),
        ("98.6.", -15.0, "Temperature ends with decimal point", "98.6"),
        ("abc", -30.0, "Invalid temperature format", None),
    ]
    
    for text, expected_adjustment, expected_reason, expected_correction in invalid_cases:
        validation = pattern_matcher.validate_temperature_format(text)
        assert not validation.is_valid, f"Should not validate {text}"
        assert validation.confidence_adjustment == expected_adjustment
        assert validation.error_reason == expected_reason
        assert validation.suggested_correction == expected_correction

def test_temperature_confidence_threshold(pattern_matcher):
    """Test that readings below confidence threshold are rejected."""
    # Test with a malformed reading that should be rejected
    text = "56./"
    confidence = 83.0  # Original confidence
    
    readings = pattern_matcher.find_readings(text, confidence)
    assert len(readings) == 0, "Should reject reading below confidence threshold"
    
    # Test with a valid reading that should pass
    text = "98.6F"
    confidence = 90.0  # Increased confidence to be above minimum threshold
    
    readings = pattern_matcher.find_readings(text, confidence)
    assert len(readings) == 1, "Should accept valid reading above threshold"
    assert readings[0].confidence >= pattern_matcher.TEMP_MIN_CONFIDENCE

def test_temperature_range_validation(pattern_matcher):
    """Test temperature range validation."""
    # Test valid temperatures
    valid_cases = [
        ("98.6F", 95.0),
        ("102.4F", 95.0),
        ("37.0C", 95.0),
        ("39.1C", 95.0),
    ]
    
    for text, confidence in valid_cases:
        readings = pattern_matcher.find_readings(text, confidence)
        assert len(readings) == 1, f"Should accept {text}"
        assert readings[0].is_valid, f"Should validate {text}"
    
    # Test invalid temperatures
    invalid_cases = [
        ("94.0F", 95.0),  # Too low
        ("106.0F", 95.0),  # Too high
        ("34.0C", 95.0),  # Too low
        ("42.0C", 95.0),  # Too high
    ]
    
    for text, confidence in invalid_cases:
        readings = pattern_matcher.find_readings(text, confidence)
        assert len(readings) == 1, f"Should detect {text}"
        assert not readings[0].is_valid, f"Should reject {text}"

def test_temperature_validation_details(pattern_matcher):
    """Test that validation details are properly included in readings."""
    # Test with a malformed reading
    text = "56./"
    confidence = 90.0  # High enough to not be rejected by threshold
    
    readings = pattern_matcher.find_readings(text, confidence)
    assert len(readings) == 0, "Should reject malformed reading"
    
    # Test with a valid reading
    text = "98.6F"
    confidence = 90.0
    
    readings = pattern_matcher.find_readings(text, confidence)
    assert len(readings) == 1, "Should accept valid reading"
    assert readings[0].validation_details is None, "Valid reading should have no validation details"
    
    # Test with a reading missing unit
    text = "98.6"
    confidence = 90.0
    
    readings = pattern_matcher.find_readings(text, confidence)
    assert len(readings) == 0, "Should reject reading without unit" 