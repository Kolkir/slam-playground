from sklearn.neighbors import NearestNeighbors
import numpy as np

from playground.utils.transform import transform_points


class ICP:
    """
        Iterative Closest Point (ICP) implementation
    """

    def __init__(self, max_iterations=20, tolerance=0.001):
        self.__knn = NearestNeighbors(n_neighbors=1)
        self.__max_iterations = max_iterations
        self.__tolerance = tolerance

    def __nearest_neighbors(self, points_a, points_b):
        estimator = self.__knn.fit(points_b)
        distances, indices = estimator.kneighbors(points_a, return_distance=True)
        return distances.ravel(), indices.ravel()

    def __fit_transform(self, points_a, points_b):
        assert points_a.shape == points_b.shape

        # move points to their centroids
        center_a = np.mean(points_a, axis=0)
        center_b = np.mean(points_b, axis=0)
        centered_a = points_a - center_a
        centered_b = points_b - center_b

        # compute the rotation matrix
        h = np.dot(centered_a.T, centered_b)
        u, s, vt = np.linalg.svd(h)
        rot = np.dot(vt.T, u.T)

        # reflection case
        if np.linalg.det(rot) < 0:
            vt *= -1
            rot = np.dot(vt.T, u.T)

        # compute translation
        pos = center_b - center_a.dot(rot.T)

        # homogeneous transformation
        tr = np.identity(3)
        tr[:, :] = rot
        tr[:2, 2] = pos[:2]

        return tr, rot, pos

    def __get_distances(self, points_a, points_b):
        assert points_a.shape == points_b.shape
        distances = np.linalg.norm(points_a - points_b, axis=1)
        return distances

    def find_transform(self, frame_a, frame_b, indices=None):
        """
        Finds relative transformation between frames scans in local coordinates
        """
        # make points homogeneous
        points_a = np.hstack([frame_a.observed_points, np.ones((frame_a.observed_points.shape[0], 1))])
        points_b = np.hstack([frame_b.observed_points, np.ones((frame_b.observed_points.shape[0], 1))])

        # hold initial points_a
        points_initial = points_a.copy()

        prev_error = 0
        for i in range(self.__max_iterations):
            if indices is None:
                distances, indices = self.__nearest_neighbors(points_a, points_b)
            else:
                distances = self.__get_distances(points_a, points_b[indices, :])
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < self.__tolerance:
                break
            else:
                tr, _, _ = self.__fit_transform(points_a, points_b[indices, :])
                points_a = points_a.dot(tr.T)
            prev_error = mean_error

        _, rot, pos = self.__fit_transform(points_initial, points_a)

        return rot, pos[:2], prev_error
