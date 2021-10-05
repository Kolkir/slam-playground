import numpy as np


def create_rotation_matrix(angle):
    mat = np.identity(3)
    cos_value = np.cos(np.radians(angle))
    sin_value = np.sin(np.radians(angle))
    mat[:2, :2] = np.array(((cos_value, -sin_value),
                            (sin_value, cos_value)))
    return mat


def to_screen_coords(h, w, pos, clip=True):
    y, x = pos
    y = h / 2 - y
    x = w / 2 + x
    if clip:
        y = np.clip(y, 0, h).astype(int)
        x = np.clip(x, 0, w).astype(int)
    return x, y


def make_direction(rotation_matrix):
    direction = np.array([[1, 0, 1]])
    direction = np.matmul(rotation_matrix, direction.T)
    direction = np.reshape(direction, (1, 3))[0, :2]
    return direction


def get_rotation_angle(rotation_matrix):
    y_direction = np.array([[1, 0, 1]])
    direction = np.array([[1, 0, 1]])
    direction = direction.dot(rotation_matrix.T)
    angle = np.clip(np.dot(y_direction, direction.T), -1, 1)
    angle = np.degrees(np.arccos(angle))
    return angle


def transform_points(points, matrix, target_type=int):
    # Convert into homogeneous coordinates
    points = np.hstack([points, np.ones((points.shape[0], 1))])
    aligned_points = points.dot(matrix.T)
    return aligned_points[:, :2].astype(target_type)
