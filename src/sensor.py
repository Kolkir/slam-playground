import math

import pygame
import numpy as np
from skimage.draw import line_aa

from src.transform import create_rotation_matrix, to_screen_coords


class Sensor:
    def __init__(self, dist_range, fov, mu, sigma):
        self.__dist_range = dist_range
        self.__fov = fov
        self.__mu = mu
        self.__sigma = sigma
        self.__obstacles = None

        # generate scan arc coordinates
        num_scan_points = 300
        theta = np.linspace(0, 2 * np.pi, num_scan_points)
        x = np.cos(theta)
        y = np.sin(theta)
        self.__circle_coords = np.stack([x, y], axis=1)

    def get_obstacles(self):
        return self.__obstacles

    def scan(self, position, direction, world):
        # do ray tracing
        obstacles_coords = []
        start_pos = position
        for circle_dir in self.__circle_coords:
            dot_product = np.dot(direction, circle_dir)
            scan_angle = math.acos(np.clip(dot_product, -1., 1))
            scan_angle = np.degrees(scan_angle)
            if scan_angle <= self.__fov / 2:
                end_pos = (start_pos + circle_dir * self.__dist_range).astype(int)
                start_pos = start_pos.astype(int)
                end_pos = end_pos.astype(int)
                ys, xs, _ = line_aa(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
                for pos in zip(ys, xs):
                    if world.is_obstacle(pos):
                        obstacles_coords.append(pos)
                        break

        if len(obstacles_coords) > 0:
            obstacles_coords = np.array(obstacles_coords)
            obstacles_coords -= start_pos

            noise_x = int(np.random.normal(self.__mu, self.__sigma))
            noise_y = int(np.random.normal(self.__mu, self.__sigma))
            obstacles_coords += np.array([[noise_y, noise_x]])

            obstacles_coords = np.clip(obstacles_coords, [-world.height // 2 + 1, -world.width // 2 + 1],
                                       [world.height // 2 - 1, world.width // 2 - 1])

            self.__obstacles = obstacles_coords
        else:
            self.__obstacles = None

    def draw(self, screen, h, w, position, direction):
        color = (128, 128, 128)
        start_pos = pygame.math.Vector2(position[0], position[1])
        dir = pygame.math.Vector2(direction[0], direction[1])
        dir = dir.rotate(self.__fov / 2)
        end_pos1 = start_pos + dir * self.__dist_range
        dir = dir.rotate(-self.__fov)
        end_pos2 = start_pos + dir * self.__dist_range
        start_pos = to_screen_coords(h, w, start_pos)
        end_pos1 = to_screen_coords(h, w, end_pos1)
        end_pos2 = to_screen_coords(h, w, end_pos2)
        pygame.draw.line(screen, color=color, start_pos=start_pos, end_pos=end_pos1, width=2)
        pygame.draw.line(screen, color=color, start_pos=start_pos, end_pos=end_pos2, width=2)
