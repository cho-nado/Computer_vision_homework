import cv2
import numpy as np
import math


def transform_coords(x, y, angle):
    """
    Transform coordinates based on rotation angle.

    :param x: x-coordinate of the point
    :param y: y-coordinate of the point
    :param angle: rotation angle in degrees
    :return: transformed coordinates (new_x, new_y)
    """
    rad_angle = math.radians(angle)
    new_x = x * math.cos(rad_angle) - y * math.sin(rad_angle)
    new_y = x * math.sin(rad_angle) + y * math.cos(rad_angle)
    return new_x, new_y

def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Rotate an image clockwise by an angle from 0 to 360 degrees and adjust the image size.

    :param image: original image
    :param point: (x, y) coordinates around which to rotate the image
    :param angle: rotation angle
    :return: rotated image
    """
    angle = -angle

    height, width, _ = image.shape
    x_min = width
    x_max = 0
    y_min = height
    y_max = 0

    # Calculate new image boundaries
    for y in range(height):
        for x in range(width):
            new_x, new_y = transform_coords(x, y, angle)

            if new_x < x_min:
                x_min = int(new_x)
            if new_x > x_max:
                x_max = int(new_x)

            if new_y < y_min:
                y_min = int(new_y)
            if new_y > y_max:
                y_max = int(new_y)

    new_height = y_max - y_min
    new_width = x_max - x_min
    rotated_image = np.zeros((new_height, new_width, 3), dtype=int)

    # Fill the rotated image
    for y in range(height):
        for x in range(width):
            new_x, new_y = transform_coords(x, y, angle)
            try:
                rotated_image[round(new_y - y_min), round(new_x - x_min)] = np.array(image[y, x], dtype=int)
            except IndexError:
                pass

    return rotated_image

def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Apply affine transformation based on the mapping of points points1 -> points2 and 
    adjust the image size.

    :param image: original image
    :param points1: source points for the transformation
    :param points2: destination points for the transformation
    :return: transformed image
    """
    transformation_matrix = cv2.getAffineTransform(points1, points2)

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Define image corners
    corners = np.array([
        [0, 0],
        [0, img_height],
        [img_width, 0],
        [img_width, img_height]
    ], dtype=np.float32)

    # Apply transformation to corners
    ones = np.ones((4, 1), dtype=np.float32)
    corners_homogeneous = np.hstack((corners, ones))
    transformed_corners = np.dot(transformation_matrix, corners_homogeneous.T).T

    # Find new image boundaries
    x_min = np.min(transformed_corners[:, 0])
    x_max = np.max(transformed_corners[:, 0])
    y_min = np.min(transformed_corners[:, 1])
    y_max = np.max(transformed_corners[:, 1])

    # Adjust transformation matrix for the offset
    transformation_matrix[0, 2] -= x_min
    transformation_matrix[1, 2] -= y_min

    # Apply affine transformation to the image
    new_width = round(x_max - x_min)
    new_height = round(y_max - y_min)
    transformed_image = cv2.warpAffine(image, transformation_matrix, (new_width, new_height))

    return transformed_image
