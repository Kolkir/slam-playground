import numpy as np
from transform import create_rotation_matrix


class Body:
    """
        Physical environment simulation element
    """
    def __init__(self):
        self.__pos = np.zeros((1, 3))
        self.__rotation_matrix = np.identity(3)

    def rotate(self, angle):
        tr_mat = create_rotation_matrix(angle)
        self.__rotation_matrix = np.matmul(self.__rotation_matrix, tr_mat)

    def move(self, dist):
        self.__pos = self.try_move(dist)

    def try_move(self, dist):
        dir = self.__get_dir()
        new_pos = self.__pos.copy()
        new_pos[0, :2] += dir[0, :2] * dist
        return new_pos

    @property
    def size(self):
        return 1

    @property
    def position(self):
        return self.__pos

    @property
    def direction(self):
        return self.__get_dir()

    @property
    def rotation(self):
        return self.__rotation_matrix

    def __get_dir(self):
        dir = np.array([[1, 0, 1]])
        dir = np.matmul(self.__rotation_matrix, dir.T)
        dir = np.reshape(dir, (1, 3))
        return dir
