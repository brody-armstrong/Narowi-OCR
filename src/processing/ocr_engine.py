import pytesseract
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

class OCREngine:
    """Wrapper for Tesseract OCR functionality."""
    
    def __init__(self, config: Optional[Dict] = None, device_type: Optional[str] = None):
        """
        Initialize OCR engine with optional configuration.
        
        Args:
            config: Dictionary of Tesseract configuration parameters
            device_type: Optional string for device type (unused, for compatibility)
        """
        self.config = config or {
            '--oem': '1',  # Use Legacy + LSTM OCR Engine Mode
            '--psm': '7',  # Treat the image as a single text line
            'tessedit_char_whitelist': '0123456789./-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',  # Allow letters and numbers
        }
        # device_type is accepted for compatibility but not used
        
    def set_psm(self, psm_mode: str) -> None:
        """
        Set the PSM (Page Segmentation Mode) for Tesseract OCR.
        
        Args:
            psm_mode: PSM mode as a string (e.g., '3', '6', '7', '8', '13')
        """
        self.config['--psm'] = psm_mode

    def get_psm(self) -> str:
        """
        Get the current PSM (Page Segmentation Mode) from Tesseract configuration.

        Returns:
            str: Current PSM mode (e.g., '3', '6', '10')
        """
        return self.config.get('--psm', '3') # Default to '3' if somehow not set

    def _get_config_string(self) -> str:
        config_str = ''
        for k, v in self.config.items():
            if k.startswith('--'):
                config_str += f'{k} {v} '
            else:
                config_str += f'-c {k}={v} '
        return config_str.strip()

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from preprocessed image.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            str: Extracted text
        """
        try:
            config_str = self._get_config_string()
            text = pytesseract.image_to_string(
                image,
                config=config_str
            )
            return text.strip()
        except Exception as e:
            print(f"Error during OCR: {e}") # Consider logging this instead of printing
            return ""

    def get_confidence(self, image: np.ndarray) -> float:
        """
        Get confidence score for OCR result.
        
        Args:
            image: Preprocessed image
            
        Returns:
            float: Confidence score (0-100)
        """
        try:
            config_str = self._get_config_string()
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=config_str
            )
            
            # Calculate average confidence for non-empty text
            confidences = [int(conf) for conf, text in zip(data['conf'], data['text']) if text.strip() and int(conf) >= 0] # Filter out -1 confidences (often for empty regions)
            return sum(confidences) / len(confidences) if confidences else 0.0
        except Exception as e:
            print(f"Error getting confidence: {e}") # Consider logging
            return 0.0

    def extract_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Extract text and confidence score from image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Tuple[str, float]: (extracted text, confidence score)
        """
        # Note: extract_text and get_confidence will each call _get_config_string.
        # If performance becomes an issue, this could be optimized by calling image_to_data once.
        text = self.extract_text(image)
        confidence = self.get_confidence(image) # This might re-process if not careful or if image_to_data not used for text
        return text, confidence

    def extract_detailed_data(self, image: np.ndarray) -> list[dict]:
        try:
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=self._get_config_string()
            )
            char_list = []
            num_items = len(data['text'])
            for i in range(num_items):
                if int(data['conf'][i]) > -1 and data['text'][i].strip(): # Only process actual characters
                    char_list.append({
                        'level': data['level'][i],
                        'page_num': data['page_num'][i],
                        'block_num': data['block_num'][i],
                        'par_num': data['par_num'][i],
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i],
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'conf': float(data['conf'][i]), # Ensure conf is float
                        'text': data['text'][i]
                    })
            return char_list
        except Exception as e:
            print(f"Error during detailed OCR data extraction: {e}")
            return []
        