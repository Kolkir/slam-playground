import unittest
import numpy as np
from playground.slam.icp import ICP
from playground.slam.frame import Frame
from playground.utils.transform import create_rotation_matrix
from playground.utils.transform import transform_points


class ICPTests(unittest.TestCase):

    def test_position(self):
        points_a = np.array([[0, 0],
                             [0, 5],
                             [0, 10],
                             [0, 15]])

        offset = np.array([2, 2])
        points_b = points_a + offset

        frame_a = Frame(np.array([0, 0, 1]), np.identity(3), points_a)
        frame_b = Frame(np.array([0, 0, 1]), np.identity(3), points_b)

        icp = ICP()
        rot, pos, error = icp.find_transform(frame_a, frame_b)
        self.assertTrue(np.allclose(offset, pos))
        self.assertTrue(np.allclose(np.identity(3), rot))

    def test_rotation_manual_index(self):
        points_a = np.array([[0, 0],
                             [0, 5],
                             [0, 10],
                             [0, 15]])

        rotation = create_rotation_matrix(-45)
        center_a = np.mean(points_a, axis=0)
        move = np.identity(3)
        move[:2, 2] = -center_a
        move_back = np.identity(3)
        move_back[:2, 2] = center_a
        tr = np.matmul(move_back, rotation)
        tr = np.matmul(tr, move)

        points_b = transform_points(points_a, tr, target_type=float)

        indices = np.array([0, 1, 2, 3])

        frame_a = Frame(np.array([0, 0, 1]), np.identity(3), points_a)
        frame_b = Frame(np.array([0, 0, 1]), np.identity(3), points_b)

        icp = ICP()
        rot, pos, error = icp.find_transform(frame_a, frame_b, indices)

        points_a = transform_points(points_a, rot, target_type=float)
        points_a = points_a.astype(float) + pos

        self.assertTrue(np.allclose(points_a, points_b))
        self.assertTrue(np.allclose(rotation, rot))

    def test_rotation(self):
        points_a = np.array([[0, 0],
                             [0, 5],
                             [0, 10],
                             [0, 15]])

        rotation = create_rotation_matrix(-45)
        center_a = np.mean(points_a, axis=0)
        move = np.identity(3)
        move[:2, 2] = -center_a
        move_back = np.identity(3)
        move_back[:2, 2] = center_a
        tr = np.matmul(move_back, rotation)
        tr = np.matmul(tr, move)

        points_b = transform_points(points_a, tr, target_type=float)

        frame_a = Frame(np.array([0, 0, 1]), np.identity(3), points_a)
        frame_b = Frame(np.array([0, 0, 1]), np.identity(3), points_b)

        icp = ICP()
        rot, pos, error = icp.find_transform(frame_a, frame_b)

        points_a = transform_points(points_a, rot, target_type=float)
        points_a = points_a + pos

        self.assertTrue(np.allclose(points_a, points_b))
        self.assertTrue(np.allclose(rotation, rot))

    def test_rotation_translation(self):
        points_a = np.array([[0, 0],
                             [0, 5],
                             [0, 10],
                             [0, 15]])

        rotation = create_rotation_matrix(-45)
        center_a = np.mean(points_a, axis=0)
        move = np.identity(3)
        move[:2, 2] = -center_a
        move_back = np.identity(3)
        move_back[:2, 2] = center_a
        move_add = np.identity(3)
        move_add[:2, 2] = [2., 2.]

        tr = np.matmul(move_add, move_back)
        tr = np.matmul(tr, rotation)
        tr = np.matmul(tr, move)

        points_b = transform_points(points_a, tr, target_type=float)

        frame_a = Frame(np.array([0, 0, 1]), np.identity(3), points_a)
        frame_b = Frame(np.array([0, 0, 1]), np.identity(3), points_b)

        # unfortunately knn search fails in the current case, so setup indices manually
        indices = np.array([0, 1, 2, 3])
        icp = ICP()
        rot, pos, error = icp.find_transform(frame_a, frame_b, indices)

        points_a = transform_points(points_a, rot, target_type=float)
        points_a = points_a + pos

        self.assertTrue(np.allclose(points_a, points_b))
        self.assertTrue(np.allclose(rotation, rot))


if __name__ == '__main__':
    unittest.main()
