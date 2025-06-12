import sys
# Add /app to sys.path to allow importing from tests.test_digit_recognition
sys.path.append('/app')
from tests.test_digit_recognition import generate_synthetic_image

images_to_generate = {
    "3": "tests/test_data/digit_confusion/img_3.png",
    "5": "tests/test_data/digit_confusion/img_5.png",
    "13": "tests/test_data/digit_confusion/img_13.png",
    "15": "tests/test_data/digit_confusion/img_15.png",
    "33": "tests/test_data/digit_confusion/img_33.png",
    "35": "tests/test_data/digit_confusion/img_35.png",
    "53": "tests/test_data/digit_confusion/img_53.png",
    "55": "tests/test_data/digit_confusion/img_55.png",
    "835": "tests/test_data/digit_confusion/img_835.png",
    "853": "tests/test_data/digit_confusion/img_853.png",
}

for text, path in images_to_generate.items():
    generate_synthetic_image(text, path)
feature/pattern-matcher-enhancements
    print(f"Generated {path}")

    print(f"Generated {path}"
          main
