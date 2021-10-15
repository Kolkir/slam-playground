import math
import unittest
import numpy as np
from playground.slam.posegraph import PoseGraph
from playground.utils.transform import create_rotation_matrix_yx


def fill_vertices(prior_pose_index, pose_graph):
    pose_graph.add_vertex(prior_pose_index + 1, tx=0.5, ty=0.0, rot=0.2)
    pose_graph.add_vertex(prior_pose_index + 2, tx=2.3, ty=0.1, rot=-0.2)
    pose_graph.add_vertex(prior_pose_index + 3, tx=4.1, ty=0.1, rot=math.pi / 2)
    pose_graph.add_vertex(prior_pose_index + 4, tx=4.0, ty=2.0, rot=math.pi)
    pose_graph.add_vertex(prior_pose_index + 5, tx=2.1, ty=2.1, rot=-math.pi / 2)


def fill_edges(prior_pose_index, pose_graph):
    pose_graph.add_factor_edge(prior_pose_index + 1, prior_pose_index + 2, 2, 0, 0)
    pose_graph.add_factor_edge(prior_pose_index + 2, prior_pose_index + 3, 2, 0, math.pi / 2)
    pose_graph.add_factor_edge(prior_pose_index + 3, prior_pose_index + 4, 2, 0, math.pi / 2)
    pose_graph.add_factor_edge(prior_pose_index + 4, prior_pose_index + 5, 2, 0, math.pi / 2)
    # Add the loop closure constraint
    pose_graph.add_factor_edge(prior_pose_index + 5, prior_pose_index + 2, 2, 0, math.pi / 2)


class PoseGraphTests(unittest.TestCase):
    def test_add_vertex(self):
        pose_graph = PoseGraph(edge_sigma_x=0.2, edge_sigma_y=0.2, edge_sigma_angle=0.1)
        prior_pose_index = pose_graph.prior_pose_index
        fill_vertices(prior_pose_index, pose_graph)

        self.assertIsNotNone(pose_graph.get_pose_at(prior_pose_index + 1))
        self.assertIsNotNone(pose_graph.get_pose_at(prior_pose_index + 2))
        self.assertIsNotNone(pose_graph.get_pose_at(prior_pose_index + 3))
        self.assertIsNotNone(pose_graph.get_pose_at(prior_pose_index + 4))
        self.assertIsNotNone(pose_graph.get_pose_at(prior_pose_index + 5))

    def test_optimization(self):
        pose_graph = PoseGraph(edge_sigma_x=0.2, edge_sigma_y=0.2, edge_sigma_angle=0.1)
        prior_pose_index = pose_graph.prior_pose_index
        fill_vertices(prior_pose_index, pose_graph)
        fill_edges(prior_pose_index,pose_graph)

        pose_graph.optimize()

        pose1 = pose_graph.get_pose_at(prior_pose_index + 1)
        self.assertIsNotNone(pose1)

    def test_optimization_zero_error(self):
        pose_graph = PoseGraph(edge_sigma_x=1.0, edge_sigma_y=1.0, edge_sigma_angle=1.0)
        prior_pose_index = pose_graph.prior_pose_index

        pose_graph.add_vertex(prior_pose_index + 1, tx=0.0, ty=0.0, rot=0.0)
        pose_graph.add_vertex(prior_pose_index + 2, tx=0.0, ty=1.0, rot=0.0)
        pose_graph.add_vertex(prior_pose_index + 3, tx=0.0, ty=2.0, rot=0.0)

        pose_graph.add_factor_edge(prior_pose_index + 1, prior_pose_index + 2, 0, 1, 0)
        pose_graph.add_factor_edge(prior_pose_index + 2, prior_pose_index + 3, 0, 1, 0)

        pose_graph.optimize()

        pose1 = pose_graph.get_vector_pose_at(prior_pose_index + 1)
        self.assertIsNotNone(pose1)
        self.assertTrue(np.allclose(pose1, [0., 0., 0.]))

        pose2 = pose_graph.get_vector_pose_at(prior_pose_index + 2)
        self.assertIsNotNone(pose2)
        self.assertTrue(np.allclose(pose2, [0., 1., 0.]))

        pose3 = pose_graph.get_vector_pose_at(prior_pose_index + 3)
        self.assertIsNotNone(pose3)
        self.assertTrue(np.allclose(pose3, [0., 2., 0.]))

    def test_optimization_small_error(self):
        pose_graph = PoseGraph(edge_sigma_x=1.0, edge_sigma_y=1.0, edge_sigma_angle=1.0)
        prior_pose_index = pose_graph.prior_pose_index

        pose_graph.add_vertex(prior_pose_index + 1, tx=0.0, ty=0.0, rot=0.0)
        pose_graph.add_vertex(prior_pose_index + 2, tx=0.0, ty=0.5, rot=0.0)
        pose_graph.add_vertex(prior_pose_index + 3, tx=0.0, ty=1.5, rot=0.0)

        pose_graph.add_factor_edge(prior_pose_index + 1, prior_pose_index + 2, 0, 1, 0)
        pose_graph.add_factor_edge(prior_pose_index + 2, prior_pose_index + 3, 0, 1, 0)

        pose_graph.optimize()

        pose1 = pose_graph.get_vector_pose_at(prior_pose_index + 1)
        pose2 = pose_graph.get_vector_pose_at(prior_pose_index + 2)
        pose3 = pose_graph.get_vector_pose_at(prior_pose_index + 3)

        self.assertTrue(np.allclose(pose1, [0., 0., 0.]))
        self.assertTrue(np.allclose(pose2, [0., 1., 0.]))
        self.assertTrue(np.allclose(pose3, [0., 2., 0.]))

    def test_rotation(self):
        pose_graph = PoseGraph(edge_sigma_x=1.0, edge_sigma_y=1.0, edge_sigma_angle=1.0)
        prior_pose_index = pose_graph.prior_pose_index

        rot = create_rotation_matrix_yx(30)
        pose_graph.add_vertex(prior_pose_index + 1, tx=0.0, ty=0.0, rot=rot)

        pose1 = pose_graph.get_vector_pose_at(prior_pose_index + 1)
        self.assertIsNotNone(pose1)