import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Find the road number that has no obstacle at the end of the path.

    :param image: original image
    :return: road number without an obstacle on it
    """
    # Ranges for gray and red colors
    gray_lower = (200, 200, 200)
    gray_upper = (220, 220, 220)
    red_lower = (230, 0, 0)
    red_upper = (255, 50, 50)
    # Masks for these colors
    gray_mask = cv2.inRange(image, gray_lower, gray_upper)
    red_mask = cv2.inRange(image, red_lower, red_upper)
    # Find contours of gray roads
    contours_gray, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Reverse the collection (for some reason, they are stored in reverse order...)
    contours_gray = contours_gray[::-1]
    road_number = None
    # Iterate over gray road contours
    for i, contour in enumerate(contours_gray):
        x, y, w, h = cv2.boundingRect(contour)
        # Check for the presence of a red rectangle on this road
        road_region = red_mask[y:y+h, x:x+w]
        if np.sum(road_region) == 0:
            road_number = i + 1
            break
    return road_number
