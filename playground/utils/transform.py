import numpy as np


def create_rotation_matrix_yx(angle_degrees):
    mat = np.identity(3)
    cos_value = np.cos(np.radians(angle_degrees))
    sin_value = np.sin(np.radians(angle_degrees))
    mat[:2, :2] = np.array(((cos_value, sin_value),
                            (-sin_value, cos_value)))
    return mat


def create_rotation_matrix_2xy(angle):
    mat = np.identity(2)
    cos_value = np.cos(angle)
    sin_value = np.sin(angle)
    mat[:, :] = np.array(((cos_value, -sin_value),
                          (sin_value, cos_value)))
    return mat


def create_rotation_matrix_2yx(angle):
    mat = np.identity(2)
    cos_value = np.cos(angle)
    sin_value = np.sin(angle)
    mat[:, :] = np.array(((cos_value, sin_value),
                          (-sin_value, cos_value)))
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


def transform_points(points, matrix, target_type=int):
    # Convert into homogeneous coordinates
    points = np.hstack([points, np.ones((points.shape[0], 1))])
    aligned_points = points.dot(matrix.T)
    return aligned_points[:, :2].astype(target_type)


def wrap_to_pi(angle):
    """
    Wraps the angle in radians into -pi to pi interval
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def t2v(tr):
    # homogeneous transformation to vector
    v = np.zeros((3, 1))
    v[:2, 0] = tr[:2, 2]
    v[2] = np.arctan2(tr[1, 0], tr[0, 0])
    return v


def v2t(v):
    # vector to homogeneous transformation
    c = np.cos(v[2])
    s = np.sin(v[2])
    tr = np.array([[c, -s, v[0]],
                   [s, c, v[1]],
                   [0, 0, 1]])
    return tr
