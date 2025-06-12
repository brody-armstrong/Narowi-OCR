import cv2
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional

class ImageProcessor:
    """Handles basic image preprocessing for OCR."""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from the given path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image or None if loading fails
        """
        try:
            return cv2.imread(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate if the image is suitable for processing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        if image is None:
            return False
        if image.size == 0:
            return False
        return True

    @staticmethod
    def resize_image(image: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            max_dimension: Maximum dimension (width or height)
            
        Returns:
            numpy.ndarray: Resized image
        """
        height, width = image.shape[:2]
        scale = min(max_dimension / width, max_dimension / height)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        return image

    @staticmethod
    def adjust_contrast_brightness(image: np.ndarray, 
                                 contrast: float = 1.2, 
                                 brightness: int = 10) -> np.ndarray:
        """
        Adjust image contrast and brightness.
        
        Args:
            image: Input image
            contrast: Contrast adjustment factor
            brightness: Brightness adjustment value
            
        Returns:
            numpy.ndarray: Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted

    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        feature/pattern-matcher-enhancements
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_gray = clahe.apply(gray)
        
        # Apply bilateral filter to remove noise while preserving edges
        filtered = cv2.bilateralFilter(clahe_gray, 9, 75, 75)
        
        # Apply unsharp masking to enhance edges
        gaussian = cv2.GaussianBlur(filtered, (0, 0), 3.0)
        unsharp_mask = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            unsharp_mask,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Increase contrast and brightness
        alpha = 1.8  # Contrast control
        beta = 15    # Brightness control
        enhanced = cv2.convertScaleAbs(cleaned, alpha=alpha, beta=beta)
        
        return enhanced
=======
        # Convert to grayscale if the image is color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image # Assume it's already grayscale

        # Apply CLAHE
        clahe_gray = ImageProcessor.apply_clahe(gray)

        # Apply Bilateral Filter
        bilateral_filtered_gray = ImageProcessor.apply_bilateral_filter(clahe_gray)

        # Apply unsharp masking
        sharpened_gray = ImageProcessor.apply_unsharp_masking(bilateral_filtered_gray)

        # Apply Gaussian blur after sharpening
        blurred_after_sharpen = cv2.GaussianBlur(sharpened_gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred_after_sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations
        final_output = ImageProcessor.apply_morph_operations(thresh)
        return final_output
      main

    @staticmethod
    def apply_clahe(gray_image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a grayscale image.

        Args:
            gray_image: Input grayscale image.

        Returns:
            Image after CLAHE application.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray_image)
        return clahe_image

    @staticmethod
    def apply_bilateral_filter(gray_image: np.ndarray) -> np.ndarray:
        """
        Applies Bilateral Filtering to a grayscale image.
        This can reduce noise while keeping edges sharp.

        Args:
            gray_image: Input grayscale image.

        Returns:
            Image after Bilateral Filtering.
        """
        # d: Diameter of each pixel neighborhood.
        # sigmaColor: Filter sigma in the color space.
        # sigmaSpace: Filter sigma in the coordinate space.
        filtered_image = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)
        return filtered_image

    @staticmethod
    def apply_unsharp_masking(gray_image: np.ndarray) -> np.ndarray:
        """
        Applies unsharp masking to a grayscale image.

        Args:
            gray_image: Input grayscale image.

        Returns:
            Sharpened grayscale image.
        """
        blurred = cv2.GaussianBlur(gray_image, (0,0), sigmaX=3, sigmaY=3)
        sharpened = cv2.addWeighted(gray_image, 1.5, blurred, -0.5, 0)
        return sharpened

    @staticmethod
    def apply_morph_operations(image: np.ndarray) -> np.ndarray:
        """
        Applies morphological opening and closing to the image.

        Args:
            image: Input image (should be binary).

        Returns:
            Image after morphological operations.
        """
        kernel = np.ones((3,3), np.uint8)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closed

    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """
        Save processed image to file.
        
        Args:
            image: Image to save
            output_path: Path where image should be saved
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False 