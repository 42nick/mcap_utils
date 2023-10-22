from pathlib import Path

import cv2


def load_image_opencv(img_path: str | Path):
    """Loads an image using OpenCV and converts it to RGB.

    Args:
        img_path: Path to the image.

    Returns:
        Loaded image.
    """

    return cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
