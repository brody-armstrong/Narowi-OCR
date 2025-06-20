import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExtractedNumber:
    """Class to hold extracted number information."""
    value: float
    unit: Optional[str]
    confidence: float
    raw_text: str

class NumberExtractor:
    """Extracts numerical values from OCR text."""
    
    # Common patterns for medical readings with units in different positions
    PATTERNS = {
        'blood_pressure': [
            r'(\d{2,3})[/-](\d{2,3})',  # 120/80
            r'(\d{2,3})[/-](\d{2,3})\s*(?:mmHg|BP)?',  # 120/80 mmHg
            r'(?:mmHg|BP)\s*(\d{2,3})[/-](\d{2,3})',  # mmHg 120/80
            r'BP:?\s*(\d{2,3})[/-](\d{2,3})',  # BP: 120/80
            r'(?:mmHg|BP)\n(\d{2,3})[/-](\d{2,3})',  # mmHg\n120/80 (unit above)
            r'(\d{2,3})[/-](\d{2,3})\n(?:mmHg|BP)',  # 120/80\nmmHg (unit below)
        ],
        'temperature': [
            r'(\d{2,3}\.\d{1,2})[°]?[FC]',  # 98.6°F
            r'[°]?[FC]\s*(\d{2,3}\.\d{1,2})',  # F98.6
            r'(\d{2,3}\.\d{1,2})[°]?[FC]\s*$',  # 98.6F at end
            r'^[°]?[FC]\s*(\d{2,3}\.\d{1,2})',  # F98.6 at start
            r'[°]?[FC]\n(\d{2,3}\.\d{1,2})',  # F\n98.6 (unit above)
            r'(\d{2,3}\.\d{1,2})\n[°]?[FC]',  # 98.6\nF (unit below)
        ],
        'weight': [
            r'(\d{2,3}\.\d{1,2})\s*(?:lbs|kg)',  # 150.5 lbs
            r'(?:lbs|kg)\s*(\d{2,3}\.\d{1,2})',  # lbs 150.5
            r'WT:?\s*(\d{2,3}\.\d{1,2})\s*(?:lbs|kg)',  # WT: 150.5 lbs
            r'(?:lbs|kg)\n(\d{2,3}\.\d{1,2})',  # lbs\n150.5 (unit above)
            r'(\d{2,3}\.\d{1,2})\n(?:lbs|kg)',  # 150.5\nlbs (unit below)
            r'(?:kg|kilos?)\n(\d{2,3}\.\d{1,2})',  # kg\n68.2 (unit above)
            r'(\d{2,3}\.\d{1,2})\n(?:kg|kilos?)',  # 68.2\nkg (unit below)
        ],
        'oxygen': [
            r'(\d{2,3})\s*%',  # 98%
            r'%\s*(\d{2,3})',  # %98
            r'SpO2:?\s*(\d{2,3})\s*%',  # SpO2: 98%
            r'O2:?\s*(\d{2,3})\s*%',  # O2: 98%
            r'%\n(\d{2,3})',  # %\n98 (unit above)
            r'(\d{2,3})\n%',  # 98\n% (unit below)
            r'SpO2:?\n(\d{2,3})\s*%',  # SpO2:\n98% (label above)
            r'O2:?\n(\d{2,3})\s*%',  # O2:\n98% (label above)
            r'SpO2:?\s*(\d{2,3})\n%',  # SpO2: 98\n% (unit below)
            r'O2:?\s*(\d{2,3})\n%',  # O2: 98\n% (unit below)
        ],
        'heart_rate': [
            r'(\d{2,3})\s*(?:BPM|HR)',  # 72 BPM
            r'(?:BPM|HR)\s*(\d{2,3})',  # BPM 72
            r'HR:?\s*(\d{2,3})',  # HR: 72
            r'(?:BPM|HR)\n(\d{2,3})',  # BPM\n72 (unit above)
            r'(\d{2,3})\n(?:BPM|HR)',  # 72\nBPM (unit below)
            r'HR:?\n(\d{2,3})',  # HR:\n72 (label above)
            r'HR:?\s*(\d{2,3})\nBPM',  # HR: 72\nBPM (unit below)
        ],
    }
    
    def __init__(self):
        """Initialize number extractor with compiled regex patterns."""
        self.compiled_patterns = {
            name: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for name, patterns in self.PATTERNS.items()
        }
    
    def extract_numbers(self, text: str, confidence: float) -> List[ExtractedNumber]:
        """
        Extract all numerical values from text.
        
        Args:
            text: OCR extracted text
            confidence: OCR confidence score
            
        Returns:
            List[ExtractedNumber]: List of extracted numbers with metadata
        """
        results = []
        
        # Normalize newlines to handle different line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Try each pattern type
        for reading_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    if reading_type == 'blood_pressure':
                        # Handle blood pressure as two numbers
                        systolic = float(match.group(1))
                        diastolic = float(match.group(2))
                        results.extend([
                            ExtractedNumber(
                                value=systolic,
                                unit='mmHg',
                                confidence=confidence,
                                raw_text=match.group(0)
                            ),
                            ExtractedNumber(
                                value=diastolic,
                                unit='mmHg',
                                confidence=confidence,
                                raw_text=match.group(0)
                            )
                        ])
                    else:
                        # Handle single number readings
                        value = float(match.group(1))
                        unit = self._get_unit(reading_type, match.group(0))
                        results.append(
                            ExtractedNumber(
                                value=value,
                                unit=unit,
                                confidence=confidence,
                                raw_text=match.group(0)
                            )
                        )
        
        return results
    
    def _get_unit(self, reading_type: str, raw_text: str) -> Optional[str]:
        """
        Determine the unit for a reading type.
        
        Args:
            reading_type: Type of medical reading
            raw_text: Raw text containing the reading
            
        Returns:
            Optional[str]: Unit string or None
        """
        unit_map = {
            'temperature': '°F' if 'F' in raw_text.upper() else '°C',
            'weight': 'lbs' if 'lbs' in raw_text.lower() else 'kg',
            'oxygen': '%',
            'heart_rate': 'BPM'
        }
        return unit_map.get(reading_type)
    
    def validate_reading(self, number: ExtractedNumber) -> bool:
        """
        Validate if the extracted number is within reasonable medical ranges.
        
        Args:
            number: Extracted number to validate
            
        Returns:
            bool: True if reading is within valid range
        """
        # Basic range validations
        ranges = {
            'mmHg': (60, 200),  # Blood pressure
            '°F': (95, 105),    # Temperature
            '°C': (35, 41),     # Temperature
            'lbs': (50, 500),   # Weight
            'kg': (20, 250),    # Weight
            '%': (70, 100),     # Oxygen saturation
            'BPM': (40, 200)    # Heart rate
        }
        
        if number.unit in ranges:
            min_val, max_val = ranges[number.unit]
            return min_val <= number.value <= max_val
        
        return True  # Unknown units are considered valid 