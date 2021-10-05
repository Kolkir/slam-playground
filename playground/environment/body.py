import numpy as np
from playground.utils.transform import create_rotation_matrix, make_direction


class Body:
    """
        Physical environment simulation element
    """

    def __init__(self):
        self.__pos = np.zeros(2)
        self.__rotation_matrix = np.identity(3)

    def rotate(self, angle):
        tr_mat = create_rotation_matrix(angle)
        self.__rotation_matrix = np.matmul(self.__rotation_matrix, tr_mat)

    def move(self, dist):
        self.__pos = self.try_move(dist)

    def try_move(self, dist):
        direction = make_direction(self.__rotation_matrix)
        new_pos = self.__pos.copy()
        new_pos += direction * dist
        return new_pos

    @property
    def size(self):
        return 1

    @property
    def position(self):
        return self.__pos

    @property
    def rotation(self):
        return self.__rotation_matrix

