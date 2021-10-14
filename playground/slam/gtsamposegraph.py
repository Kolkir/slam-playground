import gtsam
import numpy as np


class GTSAMPoseGraph:
    def __init__(self, edge_sigma_x, edge_sigma_y, edge_sigma_angle):
        self.__prior_sigma_x = 0.1
        self.__prior_sigma_y = 0.1
        self.__prior_sigma_theta = 0.05
        self.__prior_pose_index = 0
        self.__isam2 = gtsam.ISAM2()
        self.__graph = gtsam.NonlinearFactorGraph()
        self.__values = gtsam.Values()
        self.__edge_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([edge_sigma_x, edge_sigma_y, edge_sigma_angle]))
        self.define_prior()
        self.__optimization_result = None

    def define_prior(self):
        prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.__prior_sigma_x, self.__prior_sigma_y, self.__prior_sigma_theta]))
        prior_pose = gtsam.Pose2(0, 0, 0)
        prior_factor = gtsam.PriorFactorPose2(self.__prior_pose_index, prior_pose, prior_noise_model)
        self.__graph.add(prior_factor)
        self.__values.insert(self.__prior_pose_index, prior_pose)  # it's an initial poses estimate

    @property
    def prior_pose_index(self):
        return self.__prior_pose_index

    def add_vertex(self, index, tx, ty, rot):
        """
        Initial vertex(pose) estimation.
        Can come from odometry or ICP measurements
        """
        rot2 = gtsam.Rot2()
        rot2 = rot2.fromCosSin(rot[0, 0], rot[1, 0])
        pose = gtsam.Pose2(rot2, np.array([tx, ty]))
        self.__values.insert(index, pose)

    def add_factor_edge(self, vertex_index_a, vertex_index_b, tx, ty, rot):
        rot2 = gtsam.Rot2()
        rot2 = rot2.fromCosSin(rot[0, 0], rot[1, 0])
        pose = gtsam.Pose2(rot2, np.array([tx, ty]))
        factor = gtsam.BetweenFactorPose2(vertex_index_a, vertex_index_b, pose, self.__edge_noise_model)
        self.__graph.add(factor)

    def clear(self):
        self.__optimization_result = None
        self.__graph.resize(0)
        self.__values.clear()
        self.define_prior()

    def optimize(self, tolerance=1e-5, max_iterations=100):
        parameters = gtsam.GaussNewtonParams()
        parameters.setRelativeErrorTol(tolerance)
        parameters.setMaxIterations(max_iterations)
        optimizer = gtsam.GaussNewtonOptimizer(self.__graph, self.__values, parameters)
        self.__optimization_result = optimizer.optimize()

    def get_pose_at(self, index):
        assert(self.__optimization_result is not None)
        assert(index < self.__optimization_result.size())
        pose = self.__optimization_result.atPose2(index)
        tx = pose.x()
        ty = pose.y()
        rot = np.identity(3)
        rot[:2, :2] = pose.rotation().matrix()
        return tx, ty, rot  # gtsam exchanges x-y coordinates
