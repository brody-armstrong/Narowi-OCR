import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DisplayRegion:
    """Class to hold information about a detected display region."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    is_lcd: bool = False

class ROIDetector:
    """Detects regions of interest (displays) in medical device images."""
    
    def __init__(self, 
                 min_area: int = 500,
                 max_area: int = 300000,
                 aspect_ratio_range: Tuple[float, float] = (0.2, 4.0)):
        """
        Initialize ROI detector with configuration parameters.
        
        Args:
            min_area: Minimum area for a valid display region
            max_area: Maximum area for a valid display region
            aspect_ratio_range: Valid range for width/height ratio
        """
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
    
    def detect_displays(self, image: np.ndarray) -> List[DisplayRegion]:
        """
        Detect display regions in the image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List[DisplayRegion]: List of detected display regions
        """
        # Handle empty or invalid images
        if image is None or image.size == 0:
            return []
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and process contours
        display_regions = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Check if region meets criteria
            if (self.min_area <= area <= self.max_area and
                self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                
                # Calculate confidence based on shape regularity
                confidence = self._calculate_confidence(contour, area)
                
                # Check if region looks like an LCD display
                is_lcd = self._is_lcd_display(gray[y:y+h, x:x+w])
                
                display_regions.append(DisplayRegion(
                    x=x, y=y, width=w, height=h,
                    confidence=confidence,
                    is_lcd=is_lcd
                ))
        
        # Sort regions by confidence
        display_regions.sort(key=lambda r: r.confidence, reverse=True)
        
        return display_regions
    
    def _calculate_confidence(self, contour: np.ndarray, area: int) -> float:
        """
        Calculate confidence score for a detected region.
        
        Args:
            contour: Region contour
            area: Region area
            
        Returns:
            float: Confidence score (0-100)
        """
        # Calculate contour properties
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
            
        # Calculate shape regularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Calculate rectangularity
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # Combine metrics into confidence score
        confidence = (circularity * 0.3 + rectangularity * 0.7) * 100
        return min(max(confidence, 0), 100)
    
    def _is_lcd_display(self, region: np.ndarray) -> bool:
        """
        Check if a region looks like an LCD display.
        
        Args:
            region: Grayscale image region
            
        Returns:
            bool: True if region appears to be an LCD display
        """
        if region.size == 0:
            return False
        
        # Calculate histogram
        hist = cv2.calcHist([region], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Even more lenient criteria for LCD detection
        distinct_values = np.count_nonzero(hist > 0.005)  # Count significant peaks
        contrast = np.std(region) / 128.0  # Normalized contrast
        mean_intensity = np.mean(region)
        
        # LCDs are often mid-gray, high contrast, not too many distinct values
        return (distinct_values <= 40 and contrast > 0.12 and 80 < mean_intensity < 220)
    
    def draw_regions(self, image: np.ndarray, regions: List[DisplayRegion]) -> np.ndarray:
        """
        Draw detected regions on the image for visualization.
        
        Args:
            image: Original image
            regions: List of detected regions
            
        Returns:
            np.ndarray: Image with regions drawn
        """
        result = image.copy()
        
        for region in regions:
            # Draw rectangle
            color = (0, 255, 0) if region.is_lcd else (0, 0, 255)
            cv2.rectangle(
                result,
                (region.x, region.y),
                (region.x + region.width, region.y + region.height),
                color,
                2
            )
            
            # Add confidence score
            cv2.putText(
                result,
                f"{region.confidence:.1f}%",
                (region.x, region.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return result 