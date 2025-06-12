import cv2
import numpy as np
from typing import Optional, Tuple, List
from .image_processor import ImageProcessor
from .ocr_engine import OCREngine
from .pattern_matcher import PatternMatcher, MedicalReading

class ImageCapture:
    """Handles image capture and processing for medical device readings."""
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize the image capture system.
        
        Args:
            camera_id: ID of the camera to use (default: 0 for primary camera)
        """
        self.camera_id = camera_id
        self.cap = None
        self.image_processor = ImageProcessor()
        self.ocr_engine = OCREngine()
        self.pattern_matcher = PatternMatcher()
        
    def start_camera(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            return self.cap.isOpened()
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
            
    def stop_camera(self) -> None:
        """Stop the camera capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray: Captured frame or None if capture failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return frame
        
    def process_frame(self, frame: np.ndarray) -> Tuple[str, float, List[MedicalReading]]:
        """
        Process a captured frame to extract medical readings.
        
        Args:
            frame: Captured frame as numpy array
            
        Returns:
            Tuple[str, float, List[MedicalReading]]: (extracted text, confidence, list of readings)
        """
        # Preprocess the frame
        processed = self.image_processor.preprocess_for_ocr(frame)
        
        # Extract text and confidence
        text, confidence = self.ocr_engine.extract_with_confidence(processed)
        
        # Find medical readings in the text
        readings = self.pattern_matcher.find_readings(text, confidence)
        
        return text, confidence, readings
        
    def capture_and_process(self) -> Tuple[Optional[np.ndarray], str, float, List[MedicalReading]]:
        """
        Capture a frame and process it in one step.
        
        Returns:
            Tuple[Optional[np.ndarray], str, float, List[MedicalReading]]: 
                (original frame, extracted text, confidence, list of readings)
        """
        frame = self.capture_frame()
        if frame is None:
            return None, "", 0.0, []
            
        text, confidence, readings = self.process_frame(frame)
        return frame, text, confidence, readings
        
    def __enter__(self):
        """Context manager entry."""
        self.start_camera()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_camera() 