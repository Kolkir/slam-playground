import numpy as np
from playground.utils.transform import create_rotation_matrix


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
        direction = self.__get_dir()
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
    def direction(self):
        return self.__get_dir()

    @property
    def rotation(self):
        return self.__rotation_matrix

    def __get_dir(self):
        direction = np.array([[1, 0, 1]])
        direction = np.matmul(self.__rotation_matrix, direction.T)
        direction = np.reshape(direction, (1, 3))[0, :2]
        return direction
