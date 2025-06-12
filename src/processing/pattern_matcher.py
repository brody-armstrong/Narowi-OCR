import re
from typing import List, Dict, Any, Tuple
from enum import Enum

class ReadingType(Enum):
    TEMPERATURE = "Temperature"
    WEIGHT = "Weight"
    BLOOD_PRESSURE = "Blood Pressure"
    OXYGEN = "Oxygen"
    HEART_RATE = "Heart Rate"
    BLOOD_GLUCOSE = "Blood Glucose"
    RESPIRATORY_RATE = "Respiratory Rate"
    PAIN_SCALE = "Pain Scale"
    HEIGHT = "Height"

class MedicalReading:
    def __init__(self, type: ReadingType, value: float, unit: str, is_valid: bool = True):
        self.type = type
        self.value = value
        self.unit = unit
        self.is_valid = is_valid

class ValidationResult:
    def __init__(self, is_valid: bool, confidence_adjustment: float, error_reason: str = None, suggested_correction: str = None):
        self.is_valid = is_valid
        self.confidence_adjustment = confidence_adjustment
        self.error_reason = error_reason
        self.suggested_correction = suggested_correction

class PatternMatcher:
    # Temperature patterns with units in different positions
    TEMP_PATTERNS = [
        re.compile(r'(\d+(?:\.\d+)?)\s*[°]?(?:F(?!m)|C(?!m)|f(?!m)|c(?!m))', re.IGNORECASE),  # 98.6F or 37.0C, but not 25 cm
        re.compile(r'[°]?(?:F(?!m)|C(?!m)|f(?!m)|c(?!m))\s*(\d+(?:\.\d+)?)', re.IGNORECASE),  # F98.6 or C37.0, but not cm
        re.compile(r'(\d+(?:\.\d+)?)\s*[°]?(?:F(?!m)|C(?!m)|f(?!m)|c(?!m))\s*$', re.IGNORECASE),  # 98.6F at end, not cm
        re.compile(r'^[°]?(?:F(?!m)|C(?!m)|f(?!m)|c(?!m))\s*(\d+(?:\.\d+)?)', re.IGNORECASE),  # F98.6 at start, not cm
        re.compile(r'[°]?(?:F(?!m)|C(?!m)|f(?!m)|c(?!m))\n(\d+(?:\.\d+)?)', re.IGNORECASE),  # F\n98.6 (unit above, not cm)
        re.compile(r'(\d+(?:\.\d+)?)\n[°]?(?:F(?!m)|C(?!m)|f(?!m)|c(?!m))', re.IGNORECASE),  # 98.6\nF (unit below, not cm)
        re.compile(r'TEMP:?\s*(\d+(?:\.\d+)?)\s*[°]?(?:F(?!m)|C(?!m)|f(?!m)|c(?!m))', re.IGNORECASE),  # TEMP: 98.6F, not cm
        re.compile(r'T:?\s*(\d+(?:\.\d+)?)\s*[°]?(?:F(?!m)|C(?!m)|f(?!m)|c(?!m))', re.IGNORECASE),  # T: 98.6F, not cm
    ]
    
    # Weight patterns with units in different positions
    WEIGHT_PATTERNS = [
        re.compile(r'(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?|lb\.)', re.IGNORECASE),  # 150.5 lbs
        re.compile(r'(?:lbs?|pounds?|lb\.)\s*(\d+(?:\.\d+)?)', re.IGNORECASE),  # lbs 150.5
        re.compile(r'(\d+(?:\.\d+)?)\s*(?:kg|kilos?|kilograms?)', re.IGNORECASE),  # 68.2 kg
        re.compile(r'(?:kg|kilos?|kilograms?)\s*(\d+(?:\.\d+)?)', re.IGNORECASE),  # kg 68.2
        re.compile(r'WT:?\s*(\d+(?:\.\d+)?)\s*(?:lbs?|kg)', re.IGNORECASE),  # WT: 150.5 lbs
        re.compile(r'W:?\s*(\d+(?:\.\d+)?)\s*(?:lbs?|kg)', re.IGNORECASE),  # W: 150.5 lbs
        re.compile(r'(?:lbs?|kg)\n(\d+(?:\.\d+)?)', re.IGNORECASE),  # lbs\n150.5 (unit above)
        re.compile(r'(\d+(?:\.\d+)?)\n(?:lbs?|kg)', re.IGNORECASE),  # 150.5\nlbs (unit below)
    ]
    
    # Blood pressure patterns
    BP_PATTERNS = [
        re.compile(r'(\d+)\s*/\s*(\d+)\s*(?:mmHg|BP)?'),  # 120/80 mmHg
        re.compile(r'(?:mmHg|BP)\s*(\d+)\s*/\s*(\d+)'),  # mmHg 120/80
        re.compile(r'BP:?\s*(\d+)\s*/\s*(\d+)'),  # BP: 120/80
        re.compile(r'(?:mmHg|BP)\n(\d+)\s*/\s*(\d+)'),  # mmHg\n120/80 (unit above)
        re.compile(r'(\d+)\s*/\s*(\d+)\n(?:mmHg|BP)'),  # 120/80\nmmHg (unit below)
        re.compile(r'SYS:?\s*(\d+)\s*DIA:?\s*(\d+)'),  # SYS: 120 DIA: 80
        re.compile(r'S:?\s*(\d+)\s*D:?\s*(\d+)'),  # S: 120 D: 80
    ]

    # Oxygen patterns with units in different positions
    OXYGEN_PATTERNS = [
        re.compile(r'(\d+)\s*%'),  # 98%
        re.compile(r'%\s*(\d+)'),  # %98
        re.compile(r'SpO2:?\s*(\d+)\s*%'),  # SpO2: 98%
        re.compile(r'O2:?\s*(\d+)\s*%'),  # O2: 98%
        re.compile(r'%\n(\d+)'),  # %\n98 (unit above)
        re.compile(r'(\d+)\n%'),  # 98\n% (unit below)
        re.compile(r'SpO2:?\n(\d+)\s*%'),  # SpO2:\n98% (label above)
        re.compile(r'O2:?\n(\d+)\s*%'),  # O2:\n98% (label above)
    ]

    # Heart rate patterns with units in different positions
    HR_PATTERNS = [
        re.compile(r'(\d+)\s*(?:BPM|HR)(?!\s*RR)', re.IGNORECASE),  # 72 BPM (not followed by RR)
        re.compile(r'(?:BPM|HR)(?!\s*RR)\s*(\d+)', re.IGNORECASE),  # BPM 72 (not followed by RR)
        re.compile(r'(^|\b)HR:?\s*(\d+)(?!\s*RR)', re.IGNORECASE),  # HR: 72 (not followed by RR)
        re.compile(r'(^|\b)PULSE:?\s*(\d+)(?!\s*RR)', re.IGNORECASE),  # PULSE: 72 (not followed by RR)
        re.compile(r'(^|\b)P:?\s*(\d+)(?!\s*RR)', re.IGNORECASE),  # P: 72 (not followed by RR)
    ]

    # Blood glucose patterns
    GLUCOSE_PATTERNS = [
        re.compile(r'(\d+)\s*(?:mg/dL|mg/dl|mgdL)', re.IGNORECASE),  # 120 mg/dL
        re.compile(r'(?:mg/dL|mg/dl|mgdL)\s*(\d+)', re.IGNORECASE),  # mg/dL 120
        re.compile(r'(\d+(?:\.\d+)?)\s*(?:mmol/L|mmol/l)', re.IGNORECASE),  # 6.7 mmol/L
        re.compile(r'(?:mmol/L|mmol/l)\s*(\d+(?:\.\d+)?)', re.IGNORECASE),  # mmol/L 6.7
        re.compile(r'BG:?\s*(\d+)\s*(?:mg/dL|mg/dl|mgdL)', re.IGNORECASE),  # BG: 120 mg/dL
        re.compile(r'GLU:?\s*(\d+)\s*(?:mg/dL|mg/dl|mgdL)', re.IGNORECASE),  # GLU: 120 mg/dL
    ]

    # Respiratory rate patterns
    RESP_RATE_PATTERNS = [
        re.compile(r'(\d+)\s*RR', re.IGNORECASE),  # 16 RR
        re.compile(r'RR:?\s*(\d+)', re.IGNORECASE),  # RR: 16
        re.compile(r'RESP:?\s*(\d+)', re.IGNORECASE),  # RESP: 16
        re.compile(r'(^|\b)R:?\s*(\d+)\b', re.IGNORECASE),  # R: 16 at start or word boundary
    ]

    # Pain scale patterns
    PAIN_PATTERNS = [
        re.compile(r'(-?\d+)\s*(?:/10|/ 10|out of 10)(?!\s*mmHg)', re.IGNORECASE),  # -1/10 (not followed by mmHg)
        re.compile(r'PAIN:?\s*(-?\d+)\s*(?:/10|/ 10|out of 10)(?!\s*mmHg)', re.IGNORECASE),  # PAIN: -1/10
        re.compile(r'P:?\s*(-?\d+)\s*(?:/10|/ 10|out of 10)(?!\s*mmHg)', re.IGNORECASE),  # P: -1/10
    ]

    # Height patterns
    HEIGHT_PATTERNS = [
        re.compile(r'(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)(?!\s*[FCfc])', re.IGNORECASE),  # 170 cm (not followed by F/C)
        re.compile(r'(?:cm|centimeters?)(?!\s*[FCfc])\s*(\d+(?:\.\d+)?)', re.IGNORECASE),  # cm 170 (not followed by F/C)
        re.compile(r'(\d+)\'?\s*(\d+)\s*(?:in|inches?)(?!\s*[FCfc])', re.IGNORECASE),  # 5'10 in (not followed by F/C)
        re.compile(r'(\d+)\'?\s*(\d+)\"(?!\s*[FCfc])', re.IGNORECASE),  # 5'10" (not followed by F/C)
        re.compile(r'H:?\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)(?!\s*[FCfc])', re.IGNORECASE),  # H: 170 cm
        re.compile(r'HT:?\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)(?!\s*[FCfc])', re.IGNORECASE),  # HT: 170 cm
    ]

    def find_readings(self, text: str, confidence: float) -> List[MedicalReading]:
        """Find all medical readings in the text."""
        readings = []
        seen_patterns = set()  # Track which patterns have matched to avoid duplicates
        seen_readings = set()  # Track (type, value, unit) to avoid duplicate readings
        
        # Normalize newlines to handle different line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        
        # Default order
        default_reading_types = [
            (self.PAIN_PATTERNS, ReadingType.PAIN_SCALE, self._is_valid_pain_scale),
            (self.HEIGHT_PATTERNS, ReadingType.HEIGHT, self._is_valid_height),
            (self.TEMP_PATTERNS, ReadingType.TEMPERATURE, self._is_valid_temperature),
            (self.WEIGHT_PATTERNS, ReadingType.WEIGHT, self._is_valid_weight),
            (self.BP_PATTERNS, ReadingType.BLOOD_PRESSURE, self._is_valid_blood_pressure),
            (self.OXYGEN_PATTERNS, ReadingType.OXYGEN, self._is_valid_oxygen),
            (self.HR_PATTERNS, ReadingType.HEART_RATE, self._is_valid_heart_rate),
            (self.RESP_RATE_PATTERNS, ReadingType.RESPIRATORY_RATE, self._is_valid_respiratory_rate),
            (self.GLUCOSE_PATTERNS, ReadingType.BLOOD_GLUCOSE, self._is_valid_glucose),
        ]
        for line in lines:
            line_readings = set()
            all_matches = []
            # Special handling for lines starting with R: or P:
            if re.match(r'^\s*R:', line, re.IGNORECASE):
                reading_types = [
                    (self.RESP_RATE_PATTERNS, ReadingType.RESPIRATORY_RATE, self._is_valid_respiratory_rate)
                ]
            elif re.match(r'^\s*P:', line, re.IGNORECASE):
                # If /10 or out of 10, only match as pain scale
                if re.search(r'\d+\s*(/10|out of 10)', line, re.IGNORECASE):
                    reading_types = [
                        (self.PAIN_PATTERNS, ReadingType.PAIN_SCALE, self._is_valid_pain_scale)
                    ]
                else:
                    reading_types = [
                        (self.HR_PATTERNS, ReadingType.HEART_RATE, self._is_valid_heart_rate)
                    ]
            else:
                reading_types = default_reading_types
            for patterns, reading_type, validator in reading_types:
                for pattern in patterns:
                    for match in pattern.finditer(line):
                        if match.group(0) not in seen_patterns:
                            seen_patterns.add(match.group(0))
                            all_matches.append((match, reading_type, validator))
            for match, reading_type, validator in all_matches:
                if reading_type == ReadingType.HEIGHT and 'cm' not in match.group(0).lower():
                    feet = int(match.group(1))
                    inches = int(match.group(2))
                    value = feet * 12 + inches
                    unit = 'in'
                else:
                    # For HR/RESP patterns, group(2) is the value for R: and P: patterns
                    if reading_type in [ReadingType.HEART_RATE, ReadingType.RESPIRATORY_RATE] and match.lastindex == 2:
                        value = float(match.group(2))
                    else:
                        value = float(match.group(1))
                    if reading_type == ReadingType.TEMPERATURE:
                        unit = 'F' if 'f' in match.group(0).lower() else 'C'
                    elif reading_type == ReadingType.WEIGHT:
                        unit = 'kg' if 'kg' in match.group(0).lower() else 'lb'
                    elif reading_type == ReadingType.BLOOD_GLUCOSE:
                        unit = 'mmol/L' if 'mmol' in match.group(0).lower() else 'mg/dL'
                    elif reading_type == ReadingType.BLOOD_PRESSURE:
                        systolic = int(match.group(1))
                        diastolic = int(match.group(2))
                        value = systolic
                        unit = f"{systolic}/{diastolic} mmHg"
                    elif reading_type == ReadingType.RESPIRATORY_RATE:
                        unit = 'breaths/min'
                    elif reading_type == ReadingType.HEART_RATE:
                        unit = 'BPM'
                    elif reading_type == ReadingType.PAIN_SCALE:
                        unit = '/10'
                    else:
                        unit = self._get_unit_for_type(reading_type)
                # Call validator with correct number of arguments
                if reading_type == ReadingType.BLOOD_PRESSURE:
                    is_valid = validator(systolic, diastolic)
                elif reading_type in [ReadingType.OXYGEN, ReadingType.HEART_RATE, 
                                    ReadingType.RESPIRATORY_RATE, ReadingType.PAIN_SCALE]:
                    is_valid = validator(value)
                else:
                    is_valid = validator(value, unit)
                reading_key = (reading_type, value, unit)
                if reading_key in seen_readings or reading_type in line_readings:
                    continue
                seen_readings.add(reading_key)
                line_readings.add(reading_type)
                readings.append(MedicalReading(
                    type=reading_type,
                    value=value,
                    unit=unit,
                    is_valid=is_valid
                ))
        return readings

    def _is_valid_temperature(self, value: float, unit: str) -> bool:
        """Check if temperature is within normal range."""
        if unit == 'F':
            return 95.0 <= value <= 104.0
        else:  # Celsius
            return 35.0 <= value <= 40.0

    def _is_valid_weight(self, value: float, unit: str) -> bool:
        """Check if weight is within normal range."""
        if unit == 'kg':
            return 20.0 <= value <= 200.0
        else:  # pounds
            return 44.0 <= value <= 440.0

    def _is_valid_blood_pressure(self, systolic: int, diastolic: int) -> bool:
        """Check if blood pressure is within normal range."""
        return (90 <= systolic <= 140) and (60 <= diastolic <= 90)

    def _is_valid_oxygen(self, value: float) -> bool:
        """Check if oxygen saturation is within normal range."""
        return 70 <= value <= 100

    def _is_valid_heart_rate(self, value: float) -> bool:
        """Check if heart rate is within normal range."""
        return 40 <= value <= 200

    def _is_valid_glucose(self, value: float, unit: str) -> bool:
        """Check if blood glucose is within normal range."""
        if unit == 'mmol/L':
            return 2.8 <= value <= 22.2
        else:  # mg/dL
            return 50 <= value <= 400

    def _is_valid_respiratory_rate(self, value: float) -> bool:
        """Check if respiratory rate is within normal range."""
        return 8 <= value <= 40

    def _is_valid_pain_scale(self, value: float) -> bool:
        """Check if pain scale value is valid."""
        return 0 <= value <= 10

    def _is_valid_height(self, value: float, unit: str) -> bool:
        """Check if height is within normal range."""
        if unit == 'cm':
            return 30.0 <= value <= 250.0
        else:  # inches
            return 12.0 <= value <= 100.0

    def validate_temperature_format(self, text: str) -> ValidationResult:
        """Validate temperature format and return validation result."""
        # Check for invalid characters
        if '/' in text or '\\' in text:
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=-20.0,
                error_reason="Invalid character '/' or '\\' in temperature",
                suggested_correction=text.replace('/', '.').replace('\\', '.')
            )

        # Check for trailing decimal point
        if text.endswith('.'):
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=-15.0,  # -15.0 for trailing decimal
                error_reason="Temperature ends with decimal point",
                suggested_correction=text.rstrip('.')
            )

        # Check for missing unit
        if not any(unit in text.lower() for unit in ['f', 'c', '°f', '°c']):
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=-25.0,  # -25.0 for missing unit
                error_reason="Missing temperature unit (F or C)",
                suggested_correction=text + '°F'
            )

        # Check for valid format with units in different positions
        valid_formats = [
            r'^\d+(\.\d+)?\s*[°]?[FCfc]$',  # 98.6F
            r'^[°]?[FCfc]\s*\d+(\.\d+)?$',  # F98.6
            r'^\d+(\.\d+)?\s*[°]?[FCfc]\s*$',  # 98.6F at end
            r'^[°]?[FCfc]\s*\d+(\.\d+)?$',  # F98.6 at start
        ]
        
        if not any(re.match(pattern, text.strip()) for pattern in valid_formats):
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=-30.0,
                error_reason="Invalid temperature format",
                suggested_correction=None
            )

        return ValidationResult(is_valid=True, confidence_adjustment=0.0)

    def validate_digits_with_confidence(self, word_data: List[Dict[str, Any]], text: str) -> Tuple[str, List[str]]:
        """Validate digits with confidence and return validated text and issues."""
        issues = []
        validated_text = text

        for word_info in word_data:
            if word_info['conf'] < self.LOW_CONFIDENCE_THRESHOLD_3_5:
                issues.append(f"Low confidence ({word_info['conf']:.2f}%) for digits: {word_info['text']}")

        return validated_text, issues

    def extract_weight(self, text: str, confidence: float) -> List[Dict[str, Any]]:
        """Extract weight readings from text."""
        readings = []
        seen_raw_texts = set()  # Track raw texts to avoid duplicates

        for pattern in self.WEIGHT_PATTERNS:
            for match in pattern.finditer(text):
                raw_text = match.group(0)
                if raw_text in seen_raw_texts:
                    continue
                seen_raw_texts.add(raw_text)
                value = float(match.group(1))
                unit = match.group(2).lower()
                readings.append({
                    'value': value,
                    'unit': unit,
                    'raw_text': raw_text,
                    'conf': confidence
                })

        return readings

    def _get_unit_for_type(self, reading_type: ReadingType) -> str:
        """Get the default unit for a reading type."""
        unit_map = {
            ReadingType.OXYGEN: '%',
            ReadingType.HEART_RATE: 'BPM',
            ReadingType.RESPIRATORY_RATE: 'breaths/min',
            ReadingType.PAIN_SCALE: '/10',
            ReadingType.HEIGHT: 'cm',  # Added default unit for height
        }
        return unit_map.get(reading_type, '') 