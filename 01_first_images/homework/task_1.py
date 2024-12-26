import cv2
import numpy as np
from pyparsing import deque


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Find a path through the maze.

    :param image: maze image
    :return: path coordinates from the maze in the form (x, y), where x and y are arrays of coordinates
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    height, width = binary_mask.shape

    # Find the start and end points
    start_point = None
    end_point = None
    for i in range(height):
        for j in range(width):
            if binary_mask[i, j] == 0:
                start_point = (i, j)
                break
        if start_point is not None:
            break

    for i in range(height - 1, -1, -1):
        for j in range(width - 1, -1, -1):
            if binary_mask[i, j] == 0:
                end_point = (i, j)
                break
        if end_point is not None:
            break

    if start_point is None or end_point is None:
        print("Failed to find the entrance or exit")
        return None

    # Directions for movement (up, down, left, right)
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Use a queue to implement the search algorithm
    queue = deque([start_point])
    visited_nodes = set()
    visited_nodes.add(start_point)
    # Dictionary to store parent nodes
    parent_map = {start_point: None}

    while queue:
        current_node = queue.popleft()

        # Exit the loop if the end point is reached
        if current_node == end_point:
            break

        for move in movements:
            neighbor_node = (current_node[0] + move[0], current_node[1] + move[1])

            # Check image boundaries and conditions for adding the neighbor node to the queue
            if (0 <= neighbor_node[0] < height and
                0 <= neighbor_node[1] < width and
                binary_mask[neighbor_node] == 0 and
                neighbor_node not in visited_nodes):

                visited_nodes.add(neighbor_node)
                queue.append(neighbor_node)
                parent_map[neighbor_node] = current_node

    # Reconstruct the path from the end point to the start point
    coords = []
    step = end_point

    while step is not None:
        coords.append(step)
        step = parent_map.get(step)

    coords.reverse()  # Reverse the list to get the correct order of coordinates
    x_coords, y_coords = ([], [])

    if coords:
        x_coords, y_coords = zip(*coords)  # Split coordinates into x and y

    return (x_coords, y_coords)
