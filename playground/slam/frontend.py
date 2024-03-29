import numpy as np
import pygame
from skimage.draw import line_aa
from sklearn.neighbors import NearestNeighbors

from playground.slam.frame import Frame
from playground.slam.icp import ICP
from playground.utils.transform import to_screen_coords, transform_points


class FrontEnd:
    """
    Takes raw sensor data and construct graph
    """

    def __init__(self, world_h, world_w):
        self.__h = world_h
        self.__w = world_w
        self.__local_map = np.full((world_h, world_w), 255, dtype=np.uint8)

        self.__last_num_to_check = 2  # number of last key-frames to try align current one
        self.__frame_align_error = 10  # distance in pixels
        self.__icp = ICP()
        self.__frames = []
        self.__knn = NearestNeighbors(n_neighbors=1)

    def __find_frames_correspondences(self, frame_a, frame_b, min_dist=5):
        ids_a = frame_a.observed_points[:, 2]
        ids_a = ids_a.reshape((-1, 1))
        ids_b = frame_b.observed_points[:, 2]
        ids_b = ids_b.reshape((-1, 1))
        estimator = self.__knn.fit(ids_b)
        distances, indices = estimator.kneighbors(ids_a, return_distance=True)
        # remove outliers
        idx_a = []
        idx_b = []
        used_ids = set()
        for d, i_b, i_a in zip(distances, indices.reshape((-1)).tolist(), range(ids_a.shape[0])):
            if i_b not in used_ids:
                if d <= min_dist:
                    idx_a.append(i_a)
                    idx_b.append(i_b)
            used_ids.add(i_b)
        return idx_a, idx_b

    def add_key_frame(self, sensor):
        frame_candidate = self.create_new_frame(sensor)
        if frame_candidate:
            if len(self.__frames) > 0:
                # align current frame with the last one
                key_frame = self.__frames[-1]
                if not self.align_new_frame(frame_candidate, key_frame):
                    print('Failed to align frame"')
                    return False
            self.__frames.append(frame_candidate)
            return True
        else:
            return False

    def create_loop_closure(self, sensor):
        frame_candidate = self.create_new_frame(sensor)
        if frame_candidate:
            if len(self.__frames) > 0:
                # align current frame with the first one
                key_frame = self.__frames[0]
                if self.align_new_frame(frame_candidate, key_frame):
                    return frame_candidate
                else:
                    print('Failed to align frame"')
        return None

    def align_new_frame(self, frame_candidate, key_frame):
        idx_a, idx_b = self.__find_frames_correspondences(frame_candidate, key_frame)
        points_a = frame_candidate.observed_points[idx_a, :2]
        points_b = key_frame.observed_points[idx_b, :2]
        rot, pos, align_error = self.__icp.find_transform(points_a, points_b)

        if align_error <= self.__frame_align_error:
            frame_candidate.rotation = rot @ key_frame.rotation  # initial guess

            frame_candidate.position[:2] = pos
            frame_candidate.position = frame_candidate.rotation @ frame_candidate.position
            frame_candidate.position += key_frame.position  # initial guess

            frame_candidate.relative_icp_position = pos
            frame_candidate.relative_icp_rotation = rot
            return True
        else:
            return False

    def create_new_frame(self, sensor):
        obstacles = sensor.get_obstacles()
        if obstacles is not None:
            return Frame(obstacles.copy())
        return None

    def generate_local_map(self):
        """
            Combines frames into local map
        """
        # Clear the occupancy map
        self.__local_map = np.full((self.__h, self.__w), 255, dtype=np.uint8)
        prev_pos = None
        for frame in self.__frames:
            # draw position
            position = to_screen_coords(self.__h, self.__w, frame.position[:2])
            if prev_pos is not None:
                rr, cc, _ = line_aa(prev_pos[1], prev_pos[0], position[1], position[0])
                rr = np.clip(rr, 0, self.__h - 1)
                cc = np.clip(cc, 0, self.__w - 1)
                self.__local_map[rr, cc] = 0
            else:
                self.__local_map[position[1], position[0]] = 0
            prev_pos = position

            # move points into the world coordinate system
            points = frame.observed_points[:, :2]
            points = transform_points(points, frame.rotation, target_type=float)
            points += frame.position[:2]
            points = points.astype(int)

            # convert them into map coordinate system
            points[:, 0] = self.__h // 2 - points[:, 0]
            points[:, 1] += self.__w // 2
            points = np.clip(points, [0, 0],
                             [self.__h - 1, self.__w - 1])
            # draw
            self.__local_map[points[:, 0], points[:, 1]] = 0

    def draw(self, screen, offset):
        self.generate_local_map()
        transposed_map = np.transpose(self.__local_map)
        surf = pygame.surfarray.make_surface(transposed_map)
        screen.blit(surf, (offset, 0))

    def get_frames(self):
        return self.__frames
