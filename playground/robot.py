import pygame
from playground.environment.body import Body
from playground.utils.transform import to_screen_coords, make_direction


class Robot(Body):
    def __init__(self, odometry, sensor):
        super().__init__()
        self.__radius = 25
        self.__odomentry = odometry
        self.__sensor = sensor

    def rotate(self, angle, world):
        super().rotate(angle)
        self.__odomentry.track_rotate(angle)
        self.__sensor.scan(self.position, self.rotation, world)

    def move(self, dist, world):
        if world.allow_move(self.try_move(dist), self.size):
            super().move(dist)
            self.__odomentry.track_move(dist)
            self.__sensor.scan(self.position, self.rotation, world)

    @property
    def size(self):
        return self.__radius

    def draw(self, screen, h, w):
        # Draw robot in the real environment
        position = to_screen_coords(h, w, self.position)
        pygame.draw.circle(screen, color=(0, 0, 255), center=position, radius=self.__radius)
        direction = make_direction(self.rotation)
        dir_pos = self.position + direction * self.__radius * 2
        dir_pos = to_screen_coords(h, w, dir_pos)
        pygame.draw.line(screen, color=(0, 255, 0), start_pos=position, end_pos=dir_pos, width=2)
        self.__sensor.draw(screen, h, w, self.position, direction)
