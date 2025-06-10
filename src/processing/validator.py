from typing import List, Dict, NamedTuple, Optional, Tuple

ValidationIssue = NamedTuple('ValidationIssue', [
    ('original_char', Optional[str]),
    ('char_index_in_word', int),
    ('word_text', str),
    ('word_confidence', float),
    ('message', str),
    ('context_type', str)
])

class Validator:
    LOW_CONFIDENCE_THRESHOLD_3_5 = 85.0 # Placeholder

    @classmethod
    def validate_digits_with_confidence(cls, word_data: List[Dict], original_text: str, context: Optional[str] = None) -> Tuple[str, List[ValidationIssue]]:
        issues: List[ValidationIssue] = []

        for word_info in word_data:
            word_text = word_info['text']
            word_conf = float(word_info['conf']) # Ensure confidence is float

            # 1. Low confidence digit check
            for char_idx, char_in_word in enumerate(word_text):
                if (char_in_word == '3' or char_in_word == '5') and word_conf < cls.LOW_CONFIDENCE_THRESHOLD_3_5:
                    issue = ValidationIssue(
                        original_char=char_in_word,
                        char_index_in_word=char_idx,
                        word_text=word_text,
                        word_confidence=word_conf,
                        message=f"Low confidence for digit '{char_in_word}' in word '{word_text}'",
                        context_type='confidence_check'
                    )
                    issues.append(issue)

            # 2. Medical Range Checking (Basic Example)
            if context == "temperature_fahrenheit":
                try:
                    float_value = float(word_text)
                    # Simplistic range, can be expanded or made configurable
                    if float_value > 110.0 or float_value < 90.0:
                        issue = ValidationIssue(
                            original_char=None, # Not specific to a char
                            char_index_in_word=-1, # Not specific to a char index
                            word_text=word_text,
                            word_confidence=word_conf,
                            message=f"Potential out-of-range temperature: {word_text}F",
                            context_type='range_check'
                        )
                        issues.append(issue)
                except ValueError:
                    # Word is not a number, ignore for range check
                    pass

        return original_text, issues
