import math
import numpy as np

from playground.utils.transform import create_rotation_matrix_yx, create_rotation_matrix_2xy


class ICP:
    """
        Iterative Closest Point (ICP) implementation
    """

    def __init__(self, max_iterations=20, tolerance=0.001):
        self.__max_iterations = max_iterations
        self.__tolerance = tolerance

    def find_transform(self, points_a, points_b, iterations=100, tolerance=1):
        # swap x - y
        points_a = points_a[:, ::-1]
        points_b = points_b[:, ::-1]

        # initial values for tx, ty, angle
        params = np.array([0.0, 0.0, 0.0])

        for i in range(iterations):
            h_sum = np.zeros((3, 3))
            b_sum = np.zeros(3)

            # modify points with params
            angle = params[2]
            rot = create_rotation_matrix_2xy(angle)
            adjusted_points = points_a.dot(rot.T)
            adjusted_points += params[:2]

            # test if we can stop
            distances = self.__get_distances(adjusted_points, points_b)
            mean_error = np.mean(distances)
            if mean_error < tolerance:
                break

            for pa, pb, pm in zip(points_a, points_b, adjusted_points):
                # Jacobian
                j = np.array([[1, 0, -math.sin(angle) * pa[0] - math.cos(angle) * pa[1]],
                              [0, 1, math.cos(angle) * pa[0] - math.sin(angle) * pa[1]]])

                # Hessian approximation
                h = j.T @ j

                # Right hand side
                e = pm - pb
                b = j.T @ e

                # accumulate
                h_sum += h
                b_sum += b

            params_update = -np.linalg.pinv(h_sum) @ b_sum
            params += params_update

        # Calculate an error
        rot = create_rotation_matrix_2xy(params[2])
        adjusted_points = points_a.dot(rot.T)
        adjusted_points += params[:2]
        distances = self.__get_distances(adjusted_points, points_b)
        mean_error = np.mean(distances)

        # make result
        rot3 = create_rotation_matrix_yx(np.degrees(params[2]))
        pos = params[:2][::-1]

        return rot3, pos, mean_error

    def __get_distances(self, points_a, points_b):
        assert points_a.shape == points_b.shape
        distances = np.linalg.norm(points_a - points_b, axis=1)
        return distances
