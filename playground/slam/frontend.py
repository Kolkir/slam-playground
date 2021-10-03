import numpy as np
import pygame

from skimage.draw import line_aa

from frame import Frame
from icp import ICP
from playground.utils.transform import to_screen_coords


class FrontEnd:
    """
    Takes raw sensor data and construct graph
    """
    def __init__(self, world_h, world_w):
        self.__h = world_h
        self.__w = world_w
        self.__map = np.full((world_h, world_w), 255, dtype=np.uint8)
        self.__prev_pos = None

        # SLAM implementation
        self.__last_num_to_check = 2  # TODO: configure constant
        self.__frame_align_error = 0.001  # TODO: configure constant
        self.__icp = ICP()
        self.__frames = []

    def take_measurements(self, odometry, sensor):
        # TODO: replace the following drawing code with the local map rendering
        # Process odometery
        position = to_screen_coords(self.__h, self.__w, odometry.position)
        if self.__prev_pos is not None:
            rr, cc, _ = line_aa(self.__prev_pos[1], self.__prev_pos[0], position[1], position[0])
            # noise in the odometry can break coordinates
            rr = np.clip(rr, 0, self.__h - 1)
            cc = np.clip(cc, 0, self.__w - 1)
            self.__map[rr, cc] = 0
        else:
            self.__map[position[1], position[0]] = 0
        self.__prev_pos = position

        # Process sensor
        obstacles = sensor.get_obstacles()
        if obstacles is not None:
            obstacles = obstacles.copy()
            obstacles += odometry.position[:2].astype(int)
            obstacles[:, 0] = self.__h // 2 - obstacles[:, 0]
            obstacles[:, 1] += self.__w // 2
            # noise in the odometry can break coordinates
            obstacles = np.clip(obstacles, [0, 0],
                                [self.__h - 1, self.__w - 1])
            self.__map[obstacles[:, 0], obstacles[:, 1]] = 0

        # create key frame
        frame_candidate = Frame(odometry.position, odometry.rotation, sensor.get_obstacles())
        if len(self.__frames) > 0:
            # check if we can align current frame with last ones
            for key_frame in self.__frames[-self.__last_num_to_check:]:
                tr, align_error = self.__icp.find_transform(key_frame, frame_candidate)
                if align_error < self.__frame_align_error:
                    # if we were able to align - add new key frame
                    self.__frames.append(frame_candidate)
        else:
            self.__frames.append(frame_candidate)

    def draw(self, screen, offset):
        transposed_map = np.transpose(self.__map)
        surf = pygame.surfarray.make_surface(transposed_map)
        screen.blit(surf, (offset, 0))

