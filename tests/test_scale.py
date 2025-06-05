import pytest
from src.processing.pattern_matcher import PatternMatcher, ReadingType

@pytest.fixture
def pattern_matcher():
    return PatternMatcher()

def test_valid_weight_formats(pattern_matcher):
    valid_cases = [
        ("150lb", 95.0, 150.0, "lbs"),
        ("72.5kg", 95.0, 72.5, "kg"),
        ("200 lbs", 95.0, 200.0, "lbs"),
        ("99.9 kg", 95.0, 99.9, "kg"),
        ("250lb", 95.0, 250.0, "lbs"),
        ("120kgs", 95.0, 120.0, "kg"),
    ]
    for text, confidence, expected_value, expected_unit in valid_cases:
        readings = pattern_matcher.extract_weight(text, confidence)
        assert len(readings) == 1, f"Should extract from {text}"
        reading = readings[0]
        assert reading.value == expected_value
        assert reading.unit == expected_unit
        assert reading.is_valid

def test_invalid_weight_formats(pattern_matcher):
    invalid_cases = [
        ("10lb", 95.0),   # Too low
        ("600lb", 95.0), # Too high
        ("5kg", 95.0),   # Too low
        ("300kg", 95.0), # Too high
        ("abc", 95.0),   # Not a number
        ("lb", 95.0),    # No value
    ]
    for text, confidence in invalid_cases:
        readings = pattern_matcher.extract_weight(text, confidence)
        if readings:
            assert not readings[0].is_valid, f"Should reject {text}"
        else:
            assert True  # No reading extracted is also valid for invalid input

def test_weight_unit_detection(pattern_matcher):
    cases = [
        ("180lb", "lbs"),
        ("80kg", "kg"),
        ("200 lbs", "lbs"),
        ("99.9 kg", "kg"),
        ("120kgs", "kg"),
    ]
    for text, expected_unit in cases:
        readings = pattern_matcher.extract_weight(text, 95.0)
        assert len(readings) == 1
        assert readings[0].unit == expected_unit

def test_weight_multiple_readings(pattern_matcher):
    text = "150lb 70kg"
    readings = pattern_matcher.extract_weight(text, 95.0)
    assert len(readings) == 2
    units = {r.unit for r in readings}
    assert "lbs" in units and "kg" in units 