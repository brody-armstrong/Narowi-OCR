from processing.image_processor import ImageProcessor
from processing.ocr_engine import OCREngine
from processing.number_extractor import NumberExtractor
import cv2
import sys

def process_medical_image(image_path: str):
    """
    Process a medical device image and extract readings.
    
    Args:
        image_path: Path to the image file
    """
    # Initialize components
    processor = ImageProcessor()
    engine = OCREngine()
    extractor = NumberExtractor()
    
    # Load and validate image
    image = processor.load_image(image_path)
    if not processor.validate_image(image):
        print(f"Error: Invalid image at {image_path}")
        return
    
    # Preprocess image
    preprocessed = processor.preprocess_for_ocr(image)
    
    # Perform OCR
    text, confidence = engine.extract_with_confidence(preprocessed)
    print(f"\nExtracted text (confidence: {confidence:.1f}%):")
    print(text)
    
    # Extract and validate numbers
    numbers = extractor.extract_numbers(text, confidence)
    
    print("\nExtracted readings:")
    for number in numbers:
        is_valid = extractor.validate_reading(number)
        status = "✓" if is_valid else "⚠"
        print(f"{status} {number.value} {number.unit} (from: {number.raw_text})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python example.py <image_path>")
        sys.exit(1)
    
    process_medical_image(sys.argv[1]) 