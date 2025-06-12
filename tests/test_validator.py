import pytest
from src.processing.validator import Validator, ValidationIssue

def test_validate_digits_with_confidence():
    validator = Validator()
    word_data = [{"text": "835", "conf": "95.00"}]
    text, issues = validator.validate_digits_with_confidence(word_data, "835")
    assert len(issues) == 0
    assert text == "835"