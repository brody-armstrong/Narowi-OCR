import cv2
import time
import os
import numpy as np
from datetime import datetime
from src.processing.image_capture import ImageCapture
from src.processing.pattern_matcher import ReadingType

def main():
    """Main function to demonstrate image capture and reading extraction."""
    # Create captures directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    captures_dir = os.path.join("captures", f"session_{timestamp}")
    os.makedirs(captures_dir, exist_ok=True)
    
    print(f"Starting medical reading capture...")
    print(f"Captures will be saved to: {captures_dir}")
    print("Press 'q' to quit, 'c' to capture and process")
    
    with ImageCapture() as capture:
        while True:
            # Just capture frame without processing
            frame = capture.capture_frame()
            
            if frame is None:
                print("Failed to capture frame")
                continue
            
            # Display the original frame
            cv2.imshow('Medical Reading Capture', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Process the frame when capture is requested
                preprocessed = capture.image_processor.preprocess_for_ocr(frame)
                text, confidence, readings = capture.process_frame(frame)
                
                # Show both original and preprocessed images
                preprocessed_display = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
                display = np.hstack((frame, preprocessed_display))
                cv2.imshow('Capture Results (Original | Preprocessed)', display)
                cv2.waitKey(1000)  # Show for 1 second
                
                # Save both original and preprocessed frames
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                image_path = os.path.join(captures_dir, f"capture_{timestamp}.jpg")
                preprocessed_path = os.path.join(captures_dir, f"preprocessed_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)
                cv2.imwrite(preprocessed_path, preprocessed)
                print(f"\nSaved capture to {image_path}")
                print(f"Saved preprocessed image to {preprocessed_path}")
                
                # Display results
                print(f"\nExtracted Text: {text}")
                print(f"Confidence: {confidence:.2f}%")
                
                if readings:
                    print("\nDetected Readings:")
                    for reading in readings:
                        print(f"- {reading.type.value}: {reading.value} {reading.unit}")
                        if not reading.is_valid:
                            print(f"  (Invalid reading - outside normal range)")
                else:
                    print("No medical readings detected")
                
                # Save the readings to a text file
                if readings:
                    readings_path = os.path.join(captures_dir, f"readings_{timestamp}.txt")
                    with open(readings_path, 'w') as f:
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"Extracted Text: {text}\n")
                        f.write(f"Confidence: {confidence:.2f}%\n\n")
                        f.write("Detected Readings:\n")
                        for reading in readings:
                            f.write(f"- {reading.type.value}: {reading.value} {reading.unit}\n")
                            if not reading.is_valid:
                                f.write(f"  (Invalid reading - outside normal range)\n")
                    print(f"Saved readings to {readings_path}")
    
    print("\nCapture session ended")

if __name__ == "__main__":
    main() 