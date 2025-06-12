import pytest
from src.processing.pattern_matcher import PatternMatcher, ReadingType, MedicalReading, ValidationResult

@pytest.fixture
def pattern_matcher():
    return PatternMatcher()

def test_temperature_patterns(pattern_matcher):
    test_cases = [
        ("98.6F", 98.6, "F", True),
        ("37.0°C", 37.0, "C", True),
        ("F98.6", 98.6, "F", True),
        ("°C37.0", 37.0, "C", True),
        ("98.6F\n", 98.6, "F", True),
        ("\nF98.6", 98.6, "F", True),
        ("TEMP: 98.6F", 98.6, "F", True),
        ("T: 98.6F", 98.6, "F", True),
        ("105F", 105.0, "F", False),  # Too high
        ("34C", 34.0, "C", False),    # Too low
    ]
    
    for text, expected_value, expected_unit, expected_valid in test_cases:
        readings = pattern_matcher.find_readings(text, 0.95)
        assert len(readings) > 0, f"No temperature reading found in: {text}"
        reading = readings[0]
        assert reading.type == ReadingType.TEMPERATURE
        assert reading.value == expected_value
        assert reading.unit == expected_unit
        assert reading.is_valid == expected_valid

def test_weight_patterns(pattern_matcher):
    test_cases = [
        ("150.5 lbs", 150.5, "lb", True),
        ("68.2 kg", 68.2, "kg", True),
        ("lbs 150.5", 150.5, "lb", True),
        ("kg 68.2", 68.2, "kg", True),
        ("WT: 150.5 lbs", 150.5, "lb", True),
        ("W: 150.5 lbs", 150.5, "lb", True),
        ("450 lbs", 450.0, "lb", False),  # Too high
        ("15 kg", 15.0, "kg", False),     # Too low
    ]
    
    for text, expected_value, expected_unit, expected_valid in test_cases:
        readings = pattern_matcher.find_readings(text, 0.95)
        assert len(readings) > 0, f"No weight reading found in: {text}"
        reading = readings[0]
        assert reading.type == ReadingType.WEIGHT
        assert reading.value == expected_value
        assert reading.unit == expected_unit
        assert reading.is_valid == expected_valid

def test_blood_pressure_patterns(pattern_matcher):
    test_cases = [
        ("120/80 mmHg", 120, "120/80 mmHg", True),
        ("mmHg 120/80", 120, "120/80 mmHg", True),
        ("BP: 120/80", 120, "120/80 mmHg", True),
        ("SYS: 120 DIA: 80", 120, "120/80 mmHg", True),
        ("S: 120 D: 80", 120, "120/80 mmHg", True),
        ("85/60 mmHg", 85, "85/60 mmHg", False),  # Too low
        ("150/95 mmHg", 150, "150/95 mmHg", False),  # Too high
    ]
    
    for text, expected_systolic, expected_unit, expected_valid in test_cases:
        readings = pattern_matcher.find_readings(text, 0.95)
        assert len(readings) > 0, f"No blood pressure reading found in: {text}"
        reading = readings[0]
        assert reading.type == ReadingType.BLOOD_PRESSURE
        assert reading.value == expected_systolic
        assert reading.unit == expected_unit
        assert reading.is_valid == expected_valid

def test_oxygen_patterns(pattern_matcher):
    test_cases = [
        ("98%", 98, "%", True),
        ("%98", 98, "%", True),
        ("SpO2: 98%", 98, "%", True),
        ("O2: 98%", 98, "%", True),
        ("65%", 65, "%", False),  # Too low
        ("101%", 101, "%", False),  # Too high
    ]
    
    for text, expected_value, expected_unit, expected_valid in test_cases:
        readings = pattern_matcher.find_readings(text, 0.95)
        assert len(readings) > 0, f"No oxygen reading found in: {text}"
        reading = readings[0]
        assert reading.type == ReadingType.OXYGEN
        assert reading.value == expected_value
        assert reading.unit == expected_unit
        assert reading.is_valid == expected_valid

def test_heart_rate_patterns(pattern_matcher):
    test_cases = [
        ("72 BPM", 72, "BPM", True),
        ("BPM 72", 72, "BPM", True),
        ("HR: 72", 72, "BPM", True),
        ("PULSE: 72", 72, "BPM", True),
        ("P: 72", 72, "BPM", True),
        ("35 BPM", 35, "BPM", False),  # Too low
        ("205 BPM", 205, "BPM", False),  # Too high
    ]
    
    for text, expected_value, expected_unit, expected_valid in test_cases:
        readings = pattern_matcher.find_readings(text, 0.95)
        assert len(readings) > 0, f"No heart rate reading found in: {text}"
        reading = readings[0]
        assert reading.type == ReadingType.HEART_RATE
        assert reading.value == expected_value
        assert reading.unit == expected_unit
        assert reading.is_valid == expected_valid

def test_blood_glucose_patterns(pattern_matcher):
    test_cases = [
        ("120 mg/dL", 120, "mg/dL", True),
        ("mg/dL 120", 120, "mg/dL", True),
        ("6.7 mmol/L", 6.7, "mmol/L", True),
        ("mmol/L 6.7", 6.7, "mmol/L", True),
        ("BG: 120 mg/dL", 120, "mg/dL", True),
        ("GLU: 120 mg/dL", 120, "mg/dL", True),
        ("45 mg/dL", 45, "mg/dL", False),  # Too low
        ("450 mg/dL", 450, "mg/dL", False),  # Too high
        ("2.5 mmol/L", 2.5, "mmol/L", False),  # Too low
        ("25 mmol/L", 25, "mmol/L", False),  # Too high
    ]
    
    for text, expected_value, expected_unit, expected_valid in test_cases:
        readings = pattern_matcher.find_readings(text, 0.95)
        assert len(readings) > 0, f"No blood glucose reading found in: {text}"
        reading = readings[0]
        assert reading.type == ReadingType.BLOOD_GLUCOSE
        assert reading.value == expected_value
        assert reading.unit == expected_unit
        assert reading.is_valid == expected_valid

def test_respiratory_rate_patterns(pattern_matcher):
    test_cases = [
        ("16 RR", 16, "breaths/min", True),
        ("RR 16", 16, "breaths/min", True),
        ("RR: 16", 16, "breaths/min", True),
        ("RESP: 16", 16, "breaths/min", True),
        ("R: 16", 16, "breaths/min", True),
        ("6 RR", 6, "breaths/min", False),  # Too low
        ("45 RR", 45, "breaths/min", False),  # Too high
    ]
    
    for text, expected_value, expected_unit, expected_valid in test_cases:
        readings = pattern_matcher.find_readings(text, 0.95)
        assert len(readings) > 0, f"No respiratory rate reading found in: {text}"
        reading = readings[0]
        assert reading.type == ReadingType.RESPIRATORY_RATE
        assert reading.value == expected_value
        assert reading.unit == expected_unit
        assert reading.is_valid == expected_valid

def test_pain_scale_patterns(pattern_matcher):
    test_cases = [
        ("7/10", 7, "/10", True),
        ("PAIN: 7/10", 7, "/10", True),
        ("P: 7/10", 7, "/10", True),
        ("7 out of 10", 7, "/10", True),
        ("-1/10", -1, "/10", False),  # Too low
        ("11/10", 11, "/10", False),  # Too high
    ]
    
    for text, expected_value, expected_unit, expected_valid in test_cases:
        readings = pattern_matcher.find_readings(text, 0.95)
        assert len(readings) > 0, f"No pain scale reading found in: {text}"
        reading = readings[0]
        assert reading.type == ReadingType.PAIN_SCALE
        assert reading.value == expected_value
        assert reading.unit == expected_unit
        assert reading.is_valid == expected_valid

def test_height_patterns(pattern_matcher):
    test_cases = [
        ("170 cm", 170, "cm", True),
        ("cm 170", 170, "cm", True),
        ("5'10\"", 70, "in", True),  # 5 feet 10 inches = 70 inches
        ("5'10 in", 70, "in", True),
        ("H: 170 cm", 170, "cm", True),
        ("HT: 170 cm", 170, "cm", True),
        ("25 cm", 25, "cm", False),  # Too low
        ("260 cm", 260, "cm", False),  # Too high
        ("5'0\"", 60, "in", True),
        ("6'0\"", 72, "in", True),
    ]
    
    for text, expected_value, expected_unit, expected_valid in test_cases:
        readings = pattern_matcher.find_readings(text, 0.95)
        assert len(readings) > 0, f"No height reading found in: {text}"
        reading = readings[0]
        assert reading.type == ReadingType.HEIGHT
        assert reading.value == expected_value
        assert reading.unit == expected_unit
        assert reading.is_valid == expected_valid

def test_temperature_validation(pattern_matcher):
    test_cases = [
        ("98.6F", True, 0.0, None, None),
        ("98.6/", False, -20.0, "Invalid character '/' or '\\' in temperature", "98.6."),
        ("98.6", False, -25.0, "Missing temperature unit (F or C)", "98.6°F"),
        ("98.6.", False, -15.0, "Temperature ends with decimal point", "98.6"),
        ("98.6F ", True, 0.0, None, None),
        (" F98.6", True, 0.0, None, None),
    ]
    
    for text, expected_valid, expected_adjustment, expected_reason, expected_correction in test_cases:
        result = pattern_matcher.validate_temperature_format(text)
        assert result.is_valid == expected_valid
        assert result.confidence_adjustment == expected_adjustment
        if expected_reason:
            assert result.error_reason == expected_reason
        if expected_correction:
            assert result.suggested_correction == expected_correction

def test_multiple_readings_in_text(pattern_matcher):
    text = """
    Temperature: 98.6F
    Blood Pressure: 120/80 mmHg
    Oxygen: 98%
    Heart Rate: 72 BPM
    Weight: 150.5 lbs
    Height: 5'10"
    Blood Glucose: 120 mg/dL
    Respiratory Rate: 16 RR
    Pain Scale: 7/10
    """
    
    readings = pattern_matcher.find_readings(text, 0.95)
    assert len(readings) == 9
    
    # Verify each reading type is present
    reading_types = {reading.type for reading in readings}
    expected_types = {
        ReadingType.TEMPERATURE,
        ReadingType.BLOOD_PRESSURE,
        ReadingType.OXYGEN,
        ReadingType.HEART_RATE,
        ReadingType.WEIGHT,
        ReadingType.HEIGHT,
        ReadingType.BLOOD_GLUCOSE,
        ReadingType.RESPIRATORY_RATE,
        ReadingType.PAIN_SCALE
    }
    assert reading_types == expected_types

def test_invalid_readings(pattern_matcher):
    text = """
    Temperature: 105F
    Blood Pressure: 150/95 mmHg
    Oxygen: 65%
    Heart Rate: 35 BPM
    Weight: 450 lbs
    Height: 25 cm
    Blood Glucose: 45 mg/dL
    Respiratory Rate: 6 RR
    Pain Scale: 11/10
    """
    
    readings = pattern_matcher.find_readings(text, 0.95)
    print("\nFound readings:")
    for reading in readings:
        print(f"Type: {reading.type}, Value: {reading.value}, Unit: {reading.unit}, Valid: {reading.is_valid}")
    assert len(readings) == 9 