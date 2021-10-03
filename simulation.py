import pygame
import argparse

from playground.rawsensorsview import RawSensorsView
from playground.robot import Robot
from playground.odometry import Odometry
from playground.sensor import Sensor
from playground.environment.world import World


def main():
    pygame.init()
    pygame.display.set_caption('SLAM playground')
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Environmental map filename')
    args = parser.parse_args()
    world = World(args.filename)
    odometry = Odometry(mu=0, sigma=3)  # noised measurements
    sensor = Sensor(dist_range=150, fov=60, mu=0, sigma=1)  # noised measurements
    robot = Robot(odometry, sensor)
    sensorsView = RawSensorsView(world.height, world.width)
    screen = pygame.display.set_mode([world.width * 2, world.height])
    rotation_step = 15  # degrees
    moving_step = 15  # points
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    robot.rotate(-rotation_step, world)
                if event.key == pygame.K_RIGHT:
                    robot.rotate(rotation_step, world)
                if event.key == pygame.K_UP:
                    robot.move(moving_step, world)
                if event.key == pygame.K_DOWN:
                    robot.move(-moving_step, world)

                sensorsView.take_measurements(odometry, sensor)

        world.draw(screen)
        robot.draw(screen, world.height, world.width)
        sensorsView.draw(screen, world.width)
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
