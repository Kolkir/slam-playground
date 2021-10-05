import numpy as np
import pygame

from skimage.draw import line_aa
from playground.utils.transform import to_screen_coords, transform_points


class RawSensorsView:
    def __init__(self, world_h, world_w):
        self.__h = world_h
        self.__w = world_w
        self.__map = np.full((world_h, world_w), 255, dtype=np.uint8)
        self.__prev_pos = None

    def take_measurements(self, odometry, sensor):
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
            # convert points into world coordinate system
            obstacles = transform_points(obstacles, odometry.rotation)
            obstacles += odometry.position[:2].astype(int)
            obstacles[:, 0] = self.__h // 2 - obstacles[:, 0]
            obstacles[:, 1] += self.__w // 2
            # noise in the odometry can break coordinates
            obstacles = np.clip(obstacles, [0, 0],
                                [self.__h - 1, self.__w - 1])
            self.__map[obstacles[:, 0], obstacles[:, 1]] = 0

    def draw(self, screen, offset):
        transposed_map = np.transpose(self.__map)
        surf = pygame.surfarray.make_surface(transposed_map)
        screen.blit(surf, (offset, 0))
