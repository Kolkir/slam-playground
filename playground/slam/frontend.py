import numpy as np
import pygame
from skimage.draw import line_aa

from playground.slam.frame import Frame
from playground.slam.icp import ICP
from playground.utils.transform import transform_points, to_screen_coords, get_rotation_angle


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

    def take_measurements(self, odometry, sensor):
        obstacles = sensor.get_obstacles()
        if obstacles is not None:
            # create key frame
            frame_candidate = Frame(odometry.position, odometry.rotation, obstacles.copy())
            if len(self.__frames) > 0:
                # check if we can align current frame with the last ones
                for key_frame in self.__frames[:-(self.__last_num_to_check+1):-1]:
                    rot, pos, align_error = self.__icp.find_transform(frame_candidate, key_frame)
                    rot_angle = get_rotation_angle(rot)
                    # skip if position was changed < 1px and rotation < 1 degree
                    if pos.mean() < 1. and rot_angle < 1.:
                        return  # we already have the same frame
                    if align_error < self.__frame_align_error:
                        # if we were able to align - add new key frame
                        frame_candidate.position = pos
                        frame_candidate.rotation = rot
                        self.__frames.append(frame_candidate)
                        break
            else:
                self.__frames.append(frame_candidate)

    def generate_local_map(self):
        """
            Combines frames into local map
        """
        # Clear the occupancy map
        self.__local_map = np.full((self.__h, self.__w), 255, dtype=np.uint8)
        current_pos = np.array([0, 0, 1])
        prev_pos = None
        for frame in self.__frames:
            current_pos = current_pos.dot(frame.transform.T)
            # draw position
            position = to_screen_coords(self.__h, self.__w, current_pos[:2])
            if prev_pos is not None:
                rr, cc, _ = line_aa(prev_pos[1], prev_pos[0], position[1], position[0])
                rr = np.clip(rr, 0, self.__h - 1)
                cc = np.clip(cc, 0, self.__w - 1)
                self.__local_map[rr, cc] = 0
            else:
                self.__local_map[position[1], position[0]] = 0
            prev_pos = position

            # # move points into the world coordinate system
            # points = transform_points(frame.observed_points, current_rot)
            # points += current_pos
            # # convert them into map coordinate system
            # points[:, 0] = self.__h // 2 - points[:, 0]
            # points[:, 1] += self.__w // 2
            # obstacles = np.clip(points, [0, 0],
            #                     [self.__h - 1, self.__w - 1])
            # # draw
            # self.__local_map[points[:, 0], points[:, 1]] = 0

    def draw(self, screen, offset):
        self.generate_local_map()
        transposed_map = np.transpose(self.__local_map)
        surf = pygame.surfarray.make_surface(transposed_map)
        screen.blit(surf, (offset, 0))

