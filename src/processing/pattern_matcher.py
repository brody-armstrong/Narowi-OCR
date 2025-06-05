import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ReadingType(Enum):
    """Types of medical readings that can be detected."""
    BLOOD_PRESSURE = "blood_pressure"
    TEMPERATURE = "temperature"
    WEIGHT = "weight"
    OXYGEN = "oxygen"
    HEART_RATE = "heart_rate"
    UNKNOWN = "unknown"

@dataclass
class ValidationResult:
    """Class to hold temperature validation results."""
    is_valid: bool
    confidence_adjustment: float
    error_reason: Optional[str]
    suggested_correction: Optional[str]

@dataclass
class MedicalReading:
    """Class to hold information about a detected medical reading."""
    type: ReadingType
    value: float
    unit: str
    confidence: float
    raw_text: str
    is_valid: bool = True
    validation_details: Optional[str] = None

class PatternMatcher:
    """Identifies and categorizes medical readings from text."""
    # Patterns are ordered from most specific to least specific
    PATTERNS = [
        # Blood pressure (must come first)
        (ReadingType.BLOOD_PRESSURE, re.compile(r'(?:^|\s)(?:BP:)?\s*(\d{2,3})[/-](\d{2,3})(?:\s|$)', re.IGNORECASE), 'mmHg'),
        (ReadingType.BLOOD_PRESSURE, re.compile(r'(?:^|\s)SYS:\s*(\d{2,3})\s*DIA:\s*(\d{2,3})(?:\s|$)', re.IGNORECASE), 'mmHg'),

        # Temperature
        (ReadingType.TEMPERATURE, re.compile(r'(?:^|\s)(?:TEMP:)?\s*(\d{2,3}(?:\.\d{1,2})?)\s*[°]?[Ff](?=\b|\s|$)', re.IGNORECASE), '°F'),
        (ReadingType.TEMPERATURE, re.compile(r'(?:^|\s)(?:TEMP:)?\s*(\d{2,3}(?:\.\d{1,2})?)\s*[°]?[Cc](?=\b|\s|$)', re.IGNORECASE), '°C'),

        # Weight
        (ReadingType.WEIGHT, re.compile(r'(?:^|\s)(?:WT:)?\s*(\d{1,4}(?:\.\d{1,2})?)\s*(?:lb|lbs|b|Ib|Ibs)\b', re.IGNORECASE), 'lbs'),
        (ReadingType.WEIGHT, re.compile(r'(?:^|\s)(?:WT:)?\s*(\d{1,4}(?:\.\d{1,2})?)\s*(?:kg|kgs|k)\b', re.IGNORECASE), 'kg'),
        (ReadingType.WEIGHT, re.compile(r'(?:^|\s)(?:WT:)?\s*(\d{1,4}(?:\.\d{1,2})?)(?:lb|lbs|b|Ib|Ibs)\b', re.IGNORECASE), 'lbs'),
        (ReadingType.WEIGHT, re.compile(r'(?:^|\s)(?:WT:)?\s*(\d{1,4}(?:\.\d{1,2})?)(?:kg|kgs|k)\b', re.IGNORECASE), 'kg'),

        # Oxygen
        (ReadingType.OXYGEN, re.compile(r'(?:^|\s)(?:SPO2:|O2:)?\s*(\d{2,3})\s*%(?:\s|$)', re.IGNORECASE), '%'),
        (ReadingType.OXYGEN, re.compile(r'(?:^|\s)(?:SPO2:|O2:)\s*(\d{2,3})(?:\s|$)', re.IGNORECASE), '%'),

        # Heart rate
        (ReadingType.HEART_RATE, re.compile(r'(?:^|\s)(?:HR:|PULSE:)?\s*(\d{2,3})\s*(?:BPM|HR)?(?:\s|$)', re.IGNORECASE), 'BPM'),
    ]

    VALID_RANGES = {
        ReadingType.BLOOD_PRESSURE: {
            'systolic': (60, 200),
            'diastolic': (40, 120)
        },
        ReadingType.TEMPERATURE: {
            '°F': (95, 105),
            '°C': (35, 41)
        },
        ReadingType.WEIGHT: {
            'lbs': (50, 500),
            'kg': (20, 250)
        },
        ReadingType.OXYGEN: {
            '%': (70, 100)
        },
        ReadingType.HEART_RATE: {
            'BPM': (40, 200)
        }
    }

    # Minimum confidence thresholds
    TEMP_MIN_CONFIDENCE = 85.0
    WEIGHT_MIN_CONFIDENCE = 85.0

    def __init__(self):
        # For backward compatibility with tests
        self.compiled_patterns = {
            reading_type: [(p.pattern, unit) for rt, p, unit in self.PATTERNS if rt == reading_type]
            for reading_type in ReadingType if reading_type != ReadingType.UNKNOWN
        }

    def find_readings(self, text: str, confidence: float) -> List[MedicalReading]:
        if text is None:
            return []
        readings = []
        used_ranges = []  # list of (start, end) tuples
        
        # Process each pattern
        for reading_type, pattern, unit in self.PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span()
                # Skip if this match is completely contained within a previous match
                if any(s <= start and end <= e for s, e in used_ranges):
                    continue
                if reading_type == ReadingType.BLOOD_PRESSURE:
                    systolic = float(match.group(1))
                    diastolic = float(match.group(2))
                    readings.append(MedicalReading(
                        type=reading_type,
                        value=systolic,
                        unit=unit,
                        confidence=confidence,
                        raw_text=match.group(0).strip(),
                        is_valid=self._validate_reading(reading_type, systolic, unit, 'systolic')
                    ))
                    readings.append(MedicalReading(
                        type=reading_type,
                        value=diastolic,
                        unit=unit,
                        confidence=confidence,
                        raw_text=match.group(0).strip(),
                        is_valid=self._validate_reading(reading_type, diastolic, unit, 'diastolic')
                    ))
                elif reading_type == ReadingType.OXYGEN and len(match.groups()) == 2:
                    value = float(match.group(2))
                    readings.append(MedicalReading(
                        type=reading_type,
                        value=value,
                        unit=unit,
                        confidence=confidence,
                        raw_text=match.group(0).strip(),
                        is_valid=self._validate_reading(reading_type, value, unit)
                    ))
                else:
                    value = float(match.group(1))
                    is_valid = self._validate_reading(reading_type, value, unit)
                    validation_details = None
                    if reading_type == ReadingType.TEMPERATURE:
                        validation = self.validate_temperature_format(match.group(0).strip())
                        is_valid = validation.is_valid
                        validation_details = validation.error_reason
                    readings.append(MedicalReading(
                        type=reading_type,
                        value=value,
                        unit=unit,
                        confidence=confidence,
                        raw_text=match.group(0).strip(),
                        is_valid=is_valid,
                        validation_details=validation_details
                    ))
                used_ranges.append((start, end))
        print(f"Returning {len(readings)} readings: {readings}")
        return readings

    def _validate_reading(self, 
                         reading_type: ReadingType, 
                         value: float, 
                         unit: str,
                         bp_type: Optional[str] = None) -> bool:
        if reading_type not in self.VALID_RANGES:
            return True
        ranges = self.VALID_RANGES[reading_type]
        if reading_type == ReadingType.BLOOD_PRESSURE and bp_type:
            min_val, max_val = ranges[bp_type]
        else:
            min_val, max_val = ranges[unit]
        print(f"Validating {reading_type} with value {value} {unit} against range {min_val}-{max_val}")
        return min_val <= value <= max_val

    def convert_unit(self, reading: MedicalReading, target_unit: str) -> Optional[MedicalReading]:
        if reading.type == ReadingType.TEMPERATURE:
            if reading.unit == '°F' and target_unit == '°C':
                new_value = (reading.value - 32) * 5/9
                return MedicalReading(
                    type=reading.type,
                    value=round(new_value, 1),
                    unit=target_unit,
                    confidence=reading.confidence,
                    raw_text=reading.raw_text,
                    is_valid=reading.is_valid
                )
            elif reading.unit == '°C' and target_unit == '°F':
                new_value = (reading.value * 9/5) + 32
                return MedicalReading(
                    type=reading.type,
                    value=round(new_value, 1),
                    unit=target_unit,
                    confidence=reading.confidence,
                    raw_text=reading.raw_text,
                    is_valid=reading.is_valid
                )
        elif reading.type == ReadingType.WEIGHT:
            if reading.unit == 'lbs' and target_unit == 'kg':
                new_value = reading.value * 0.453592
                return MedicalReading(
                    type=reading.type,
                    value=round(new_value, 1),
                    unit=target_unit,
                    confidence=reading.confidence,
                    raw_text=reading.raw_text,
                    is_valid=reading.is_valid
                )
            elif reading.unit == 'kg' and target_unit == 'lbs':
                new_value = reading.value * 2.20462
                return MedicalReading(
                    type=reading.type,
                    value=round(new_value, 1),
                    unit=target_unit,
                    confidence=reading.confidence,
                    raw_text=reading.raw_text,
                    is_valid=reading.is_valid
                )
        return None 

    def extract_weight(self, text: str, confidence: float) -> List[MedicalReading]:
        """
        Extract weight readings from text.
        """
        if confidence < self.WEIGHT_MIN_CONFIDENCE:
            return []
        readings = []
        seen_raw_texts = set()  # Track raw texts to avoid duplicates
        for reading_type, pattern, unit in self.PATTERNS:
            if reading_type != ReadingType.WEIGHT:
                continue
            for match in pattern.finditer(text):
                raw_text = match.group(0).strip()
                if raw_text in seen_raw_texts:
                    continue
                seen_raw_texts.add(raw_text)
                value = float(match.group(1))
                is_valid = self._validate_reading(reading_type, value, unit)
                readings.append(MedicalReading(
                    type=reading_type,
                    value=value,
                    unit=unit,
                    confidence=confidence,
                    raw_text=raw_text,
                    is_valid=is_valid,
                    validation_details=None if is_valid else f"Weight value {value} {unit} is out of valid range"
                ))
        return readings

    def validate_temperature_format(self, text: str) -> ValidationResult:
        """
        Validate temperature format and suggest corrections.
        """
        import re
        # Check for invalid characters
        if '/' in text or '\\' in text:
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=-20.0,
                error_reason="Invalid character '/' or '\\' in temperature",
                suggested_correction=text.replace('/', '.').replace('\\', '.')
            )
        # Check for trailing decimal
        if text.strip().endswith('.'):
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=-15.0,
                error_reason="Temperature ends with decimal point",
                suggested_correction=text.rstrip('.')
            )
        # Use regex to extract value and unit
        match = re.search(r'([\d.]+)\s*[°]?[FfCc]', text)
        if not match:
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=-30.0,
                error_reason="Invalid temperature format",
                suggested_correction=None
            )
        try:
            value = float(match.group(1))
            unit = None
            if 'F' in text.upper():
                unit = '°F'
            elif 'C' in text.upper():
                unit = '°C'
            if unit == '°F':
                if not (95 <= value <= 105):
                    return ValidationResult(
                        is_valid=False,
                        confidence_adjustment=-30.0,
                        error_reason="Temperature value out of reasonable range",
                        suggested_correction=None
                    )
            elif unit == '°C':
                if not (35 <= value <= 41):
                    return ValidationResult(
                        is_valid=False,
                        confidence_adjustment=-30.0,
                        error_reason="Temperature value out of reasonable range",
                        suggested_correction=None
                    )
            return ValidationResult(
                is_valid=True,
                confidence_adjustment=0.0,
                error_reason=None,
                suggested_correction=None
            )
        except ValueError:
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=-30.0,
                error_reason="Invalid temperature format",
                suggested_correction=None
            ) 