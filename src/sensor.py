import pygame
import numpy as np
from skimage.draw import line_aa


class Sensor:
    def __init__(self, radius, mu, sigma):
        self.__radius = radius
        self.__mu = mu
        self.__sigma = sigma
        self.__obstacles = None

        # generate scan circle coordinates
        num_scan_points = 100
        theta = np.linspace(0, 2 * np.pi, num_scan_points)
        x = self.__radius * np.cos(theta)
        y = self.__radius * np.sin(theta)
        self.__circle_coords = list(zip(x.astype(int), y.astype(int)))

    def get_obstacles(self):
        return self.__obstacles

    def scan(self, position, world):
        # do ray tracing
        obstacles_coords = []
        start_pos = position[0, :2].astype(int)
        for end_pos in self.__circle_coords:
            ys, xs, _ = line_aa(start_pos[0], start_pos[1], start_pos[0] + end_pos[0], start_pos[1] + end_pos[1])
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

    def draw(self, screen, position):
        pygame.draw.circle(screen, color=(128, 128, 128), center=position, radius=self.__radius, width=2)
