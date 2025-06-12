import pytest
import cv2
import numpy as np
import os
from src.processing.image_capture import ImageCapture
from src.processing.pattern_matcher import ReadingType

# Assuming synthetic images are generated in this path
SYNTHETIC_IMAGES_DIR = "synthetic_images/generated_readings"

def test_image_capture_initialization():
    """Test image capture initialization."""
    capture = ImageCapture()
    assert capture.camera_id == 0
    assert capture.cap is None
    assert capture.image_processor is not None
    assert capture.ocr_engine is not None
    assert capture.pattern_matcher is not None

def test_camera_control():
    """Test camera start/stop functionality."""
    capture = ImageCapture()
    
    # Test camera start
    assert capture.start_camera()
    assert capture.cap is not None
    assert capture.cap.isOpened()
    
    # Test camera stop
    capture.stop_camera()
    assert capture.cap is None

def test_context_manager():
    """Test context manager functionality."""
    with ImageCapture() as capture:
        assert capture.cap is not None
        assert capture.cap.isOpened()
    
    # Camera should be closed after context manager
    assert capture.cap is None

def test_frame_capture():
    """Test frame capture functionality."""
    with ImageCapture() as capture:
        frame = capture.capture_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3  # Should be a color image

def test_frame_processing():
    """Test frame processing functionality using a synthetic image."""
    # Load a synthetic test image with a medical reading
    image_path = os.path.join(SYNTHETIC_IMAGES_DIR, "reading_0_98.6F.png")
    test_image = cv2.imread(image_path)
    assert test_image is not None, f"Synthetic image not found at {image_path}"
    
    capture = ImageCapture()
    text, confidence, readings = capture.process_frame(test_image)
    
    print(f"Test Image OCR Text: '{text}'")
    print(f"Test Image OCR Confidence: {confidence}")

    # Check that text was extracted
    assert text
    assert confidence > 0
    
    # Check that readings were found
    assert len(readings) > 0
    assert any(r.type == ReadingType.TEMPERATURE for r in readings)

def test_capture_and_process():
    """Test combined capture and process functionality."""
    with ImageCapture() as capture:
        frame, text, confidence, readings = capture.capture_and_process()
        
        # Frame should be captured
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        
        # Text and confidence should be extracted
        assert isinstance(text, str)
        assert isinstance(confidence, float)
        
        # Readings should be a list
        assert isinstance(readings, list) 