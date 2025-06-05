import cv2
import numpy as np

def create_thermometer_image(text, filename):
    # Create a blank white image
    img = np.ones((128, 256, 3), dtype=np.uint8) * 255

    # Choose a font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get boundary of the text
    textsize = cv2.getTextSize(text, font, 3, 8)[0]

    # Center the text
    textX = (img.shape[1] - textsize[0]) // 2
    textY = (img.shape[0] + textsize[1]) // 2

    # Put black text on the image
    cv2.putText(img, text, (textX, textY), font, 3, (0, 0, 0), 8, cv2.LINE_AA)

    # Optionally, add a rectangle to simulate a display
    cv2.rectangle(img, (10, 10), (img.shape[1]-10, img.shape[0]-10), (0,0,0), 2)

    # Save the image
    cv2.imwrite(filename, img)

# Example usages:
create_thermometer_image("98.6", "thermometer_synth_1.png")
create_thermometer_image("102.4", "thermometer_synth_2.png")
create_thermometer_image("36.7", "thermometer_synth_3.png")