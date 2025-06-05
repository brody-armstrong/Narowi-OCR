import os
import pytest
import cv2
import numpy as np
from src.processing.image_processor import ImageProcessor
from src.processing.ocr_engine import OCREngine

def preprocess_image(image):
    """
    Enhanced preprocessing pipeline for real-world medical device photos.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image ready for OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size (must be odd)
        2    # C constant
    )
    
    # Find contours to detect the display area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (should be the display)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding around the detected region
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        # Crop to the display area
        cropped = thresh[y:y+h, x:x+w]
    else:
        cropped = thresh
    
    # Resize to a standard size for better OCR
    target_height = 200
    aspect_ratio = cropped.shape[1] / cropped.shape[0]
    target_width = int(target_height * aspect_ratio)
    resized = cv2.resize(cropped, (target_width, target_height), 
                        interpolation=cv2.INTER_CUBIC)
    
    # Apply morphological operations to clean up the text
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def test_ocr_real_world():
    """Test OCR extraction from real-world medical device photos."""
    # Initialize the image processor and OCR engine
    processor = ImageProcessor()
    engine = OCREngine()
    
    # TODO: Add real-world test images
    # For now, we'll use the synthetic images to test the enhanced preprocessing
    image_paths = [
        os.path.join(os.path.dirname(__file__), 'thermometer_synth_1.png'),
        os.path.join(os.path.dirname(__file__), 'thermometer_synth_2.png'),
        os.path.join(os.path.dirname(__file__), 'thermometer_synth_3.png')
    ]
    
    print("\n=== Real-World OCR Extraction Results ===")
    for image_path in image_paths:
        # Load image
        image = processor.load_image(image_path)
        assert image is not None, f"Failed to load image: {image_path}"
        
        # Apply enhanced preprocessing
        preprocessed = preprocess_image(image)
        
        # Try different PSM modes for better accuracy
        psm_modes = ['7', '8', '13']  # Single line, single word, raw line
        best_text = ""
        best_confidence = 0.0
        
        for psm in psm_modes:
            engine.set_psm(psm)
            text, confidence = engine.extract_with_confidence(preprocessed)
            
            if confidence > best_confidence:
                best_text = text
                best_confidence = confidence
        
        # Print results
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"Expected value: {os.path.basename(image_path).replace('thermometer_synth_', '').replace('.png', '')}")
        print(f"Extracted value: {best_text}")
        print(f"Confidence: {best_confidence:.2f}%")
        print("-" * 50)
        
        # Assert that text was extracted
        assert best_text, f"No text was extracted from the image: {image_path}"
        
        # Assert that the confidence is above a certain threshold
        assert best_confidence > 0.5, f"OCR confidence is too low for image: {image_path}" 