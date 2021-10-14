import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from playground.utils.transform import wrap_to_pi, v2t, t2v


class PoseGraph:
    def __init__(self, edge_sigma_x, edge_sigma_y, edge_sigma_angle):
        self.__prior_pose_index = 0
        self.__factors = []
        self.__values = dict()
        # an information matrix is the inverse of a covariance matrix
        self.__edge_noise_model = np.diag([edge_sigma_y, edge_sigma_x, edge_sigma_angle])
        self.__edge_noise_model = np.linalg.inv(self.__edge_noise_model)

    @property
    def prior_pose_index(self):
        return self.__prior_pose_index

    def add_vertex(self, index, tx, ty, rot):
        """
        Initial vertex(pose) estimation.
        Can come from odometry or ICP measurements
        """
        self.__values[index] = np.array([tx, ty, wrap_to_pi(rot)])

    def add_factor_edge(self, vertex_index_a, vertex_index_b, tx, ty, rot, noise_model=None):
        if noise_model is None:
            noise_model = self.__edge_noise_model
        factor = vertex_index_a, vertex_index_b, np.array([tx, ty, wrap_to_pi(rot)]), noise_model
        self.__factors.append(factor)

    def clear(self):
        self.__factors.clear()
        self.__values.clear()

    def __compute_error(self, value_index_i, value_index_j, factor_transform):
        # value & factor_transform : ty, tx, rot
        # Create transformation matrices from vectors
        t_i = v2t(self.__values[value_index_i])
        t_j = v2t(self.__values[value_index_j])
        t_z = v2t(factor_transform)

        # Calculate error vector
        error = t2v(np.linalg.inv(t_z) @ (np.linalg.inv(t_i) @ t_j))
        return error

    def __compute_jacobian(self, value_index_i, value_index_j, factor_transform):
        v_i = self.__values[value_index_i]
        v_j = self.__values[value_index_j]
        si = np.sin(v_i[2])
        ci = np.cos(v_i[2])
        dr_i = np.array([[-si, ci], [-ci, -si]]).T
        dt_ij = np.array([v_j[:2] - v_i[:2]]).T

        t_i = v2t(v_i)
        t_z = v2t(factor_transform)
        r_i = t_i[:2, :2]
        r_z = t_z[:2, :2]

        a_ij = np.vstack((np.hstack((-r_z.T @ r_i.T, (r_z.T @ dr_i.T) @ dt_ij)),
                          [0, 0, -1]))

        b_ij = np.vstack((np.hstack((r_z.T @ r_i.T, np.zeros((2, 1)))),
                          [0, 0, 1]))
        return a_ij, b_ij

    def optimize(self, tolerance=1e-5, iterations=100):
        num_params = 3  # tx, ty, rot
        for _ in range(iterations):
            # Building the Linear system

            num_values = len(self.__values)
            # define the normal equation matrix
            h = scipy.sparse.csc_matrix((num_values * num_params, num_values * num_params))

            # define the coefficient vector
            b = scipy.sparse.csc_matrix((num_values * num_params, 1))

            for factor in self.__factors:
                value_index_i, value_index_j, factor_transform, factor_noise_model = factor

                # compute the error
                e = self.__compute_error(value_index_i, value_index_j, factor_transform)

                # compute Jacobian parts
                a_ij, b_ij = self.__compute_jacobian(value_index_i, value_index_j, factor_transform)

                # prepare H and b blocks
                h_ii = a_ij.T @ factor_noise_model @ a_ij
                h_ij = a_ij.T @ factor_noise_model @ b_ij
                h_jj = b_ij.T @ factor_noise_model @ b_ij
                b_i = -a_ij.T @ factor_noise_model @ e
                b_j = -b_ij.T @ factor_noise_model @ e

                def id2index(value_id):
                    value_id -= 1
                    return slice((num_params * value_id), (num_params * (value_id + 1)))

                # update the coefficient vector
                b[id2index(value_index_i)] += b_i
                b[id2index(value_index_j)] += b_j

                # update the normal equation matrices
                h[id2index(value_index_i), id2index(value_index_i)] += h_ii
                h[id2index(value_index_i), id2index(value_index_j)] += h_ij
                h[id2index(value_index_j), id2index(value_index_i)] += h_ij.T
                h[id2index(value_index_j), id2index(value_index_j)] += h_jj

            # Solving the linear system

            # The system (H b) is built only from relative constraints so H is not full rank.
            # So we fix the position of the 1st vertex
            h[:num_params, :num_params] += np.eye(3)

            values_update = -scipy.sparse.linalg.spsolve(h, b)
            values_update[np.isnan(values_update)] = 0
            values_update = np.reshape(values_update, (len(self.__values), num_params))

            self.__update_values(values_update)

            # compute a mean error
            mean_error = 0
            for factor in self.__factors:
                value_index_i, value_index_j, factor_transform, _ = factor
                mean_error += self.__compute_error(value_index_i, value_index_j, factor_transform)
            mean_error /= len(self.__factors)

            # check if we converged
            if (mean_error <= tolerance).all():
                break

    def __update_values(self, values_update):
        for id, update in enumerate(values_update):
            self.__values[id + 1] += update
        pass

    def get_pose_at(self, index):
        return self.__values[index]
