import numpy as np


def create_rotation_matrix(angle):
    mat = np.identity(3)
    cos_value = np.cos(np.radians(angle))
    sin_value = np.sin(np.radians(angle))
    mat[:2, :2] = np.array(((cos_value, -sin_value),
                            (sin_value, cos_value)))
    return mat


def to_screen_coords(h, w, pos):
    y, x = pos
    y = h / 2 - y
    x = w / 2 + x
    y = np.clip(y, 0, h).astype(int)
    x = np.clip(x, 0, w).astype(int)
    return x, y

