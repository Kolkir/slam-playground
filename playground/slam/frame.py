import numpy as np


class Frame:
    def __init__(self, observed_points):
        self.position = np.array([0., 0., 0.])
        self.rotation = np.identity(3)
        self.relative_icp_position = np.array([0., 0., 0.])  # relative to the previous frame
        self.relative_icp_rotation = np.identity(3)  # relative to the previous frame
        self.__observed_points = observed_points

    @property
    def transform(self):
        # homogeneous transformation
        tr = np.identity(3)
        tr[:, :] = self.relative_icp_rotation
        tr[:2, 2] = self.relative_icp_position[:2]
        return tr

    @property
    def observed_points(self):
        return self.__observed_points
