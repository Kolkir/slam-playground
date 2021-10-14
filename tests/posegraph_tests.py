import math
import unittest
import numpy as np
from playground.slam.posegraph import PoseGraph


def fill_vertices(prior_pose_index, pose_graph):
    pose_graph.add_vertex(prior_pose_index + 1, tx=0.5, ty=0.0, rot=0.2)
    pose_graph.add_vertex(prior_pose_index + 2, tx=2.3, ty=0.1, rot=-0.2)
    pose_graph.add_vertex(prior_pose_index + 3, tx=4.1, ty=0.1, rot=math.pi / 2)
    pose_graph.add_vertex(prior_pose_index + 4, tx=4.0, ty=2.0, rot=math.pi)
    pose_graph.add_vertex(prior_pose_index + 5, tx=2.1, ty=2.1, rot=-math.pi / 2)


def fill_edges(pose_graph):
    pose_graph.add_factor_edge(1, 2, 2, 0, 0)
    pose_graph.add_factor_edge(2, 3, 2, 0, math.pi / 2)
    pose_graph.add_factor_edge(3, 4, 2, 0, math.pi / 2)
    pose_graph.add_factor_edge(4, 5, 2, 0, math.pi / 2)
    # Add the loop closure constraint
    pose_graph.add_factor_edge(5, 2, 2, 0, math.pi / 2)


class PoseGraphTests(unittest.TestCase):
    def test_add_vertex(self):
        pose_graph = PoseGraph(edge_sigma_x=0.2, edge_sigma_y=0.2, edge_sigma_angle=0.1)
        prior_pose_index = pose_graph.prior_pose_index
        fill_vertices(prior_pose_index, pose_graph)

        self.assertIsNotNone(pose_graph.get_pose_at(1))
        self.assertIsNotNone(pose_graph.get_pose_at(2))
        self.assertIsNotNone(pose_graph.get_pose_at(3))
        self.assertIsNotNone(pose_graph.get_pose_at(4))
        self.assertIsNotNone(pose_graph.get_pose_at(5))

    def test_optimization(self):
        pose_graph = PoseGraph(edge_sigma_x=0.2, edge_sigma_y=0.2, edge_sigma_angle=0.1)
        prior_pose_index = pose_graph.prior_pose_index
        fill_vertices(prior_pose_index, pose_graph)
        fill_edges(pose_graph)

        pose_graph.optimize()

        pose1 = pose_graph.get_pose_at(1)
        self.assertIsNotNone(pose1)

