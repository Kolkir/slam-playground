import numpy as np


class Frame:
    def __init__(self, position, rotation, observed_points):
        self.position = position  # relative to the previous frame
        self.rotation = rotation  # relative to the previous frame
        self.__observed_points = observed_points

    @property
    def transform(self):
        # homogeneous transformation
        tr = np.identity(3)
        tr[:, :] = self.rotation
        tr[:2, 2] = self.position[:2]
        return tr

    @property
    def observed_points(self):
        return self.__observed_points
