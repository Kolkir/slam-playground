import pygame
import argparse
from enum import Enum

import playground.slam.frontend
from playground.rawsensorsview import RawSensorsView
from playground.robot import Robot
from playground.odometry import Odometry
from playground.sensor import Sensor
from playground.environment.world import World


class SimulationMode(Enum):
    RAW_SENSORS = 1,
    ICP_ADJUSTMENT = 2,
    ICP_POSE_GRAPH_ADJUSTMENT = 3


def main():
    pygame.init()
    pygame.display.set_caption('SLAM playground')
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Environmental map filename')
    args = parser.parse_args()

    # Create simulation objects
    world = World(args.filename)
    odometry = Odometry(mu=0, sigma=0)  # noised measurements
    sensor = Sensor(dist_range=200, fov=250, mu=0, sigma=0)  # noised measurements
    robot = Robot(odometry, sensor)
    sensors_view = RawSensorsView(world.height, world.width)
    slam_front_end = playground.slam.frontend.FrontEnd(world.height, world.width)

    # Initialize rendering
    screen = pygame.display.set_mode([world.width * 2, world.height])

    # Robot movement configuration, steps should me small enough make ICP works normally
    rotation_step = 1  # degrees
    moving_step = 5  # points

    # make first initialization
    robot.move(0, world)
    sensors_view.take_measurements(odometry, sensor)
    slam_front_end.take_measurements(odometry, sensor)

    # start simulation loop
    simulation_mode = SimulationMode.RAW_SENSORS
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    simulation_mode = SimulationMode.RAW_SENSORS
                if event.key == pygame.K_i:
                    simulation_mode = SimulationMode.ICP_ADJUSTMENT
                if event.key == pygame.K_s:
                    simulation_mode = SimulationMode.ICP_POSE_GRAPH_ADJUSTMENT
                if event.key == pygame.K_LEFT:
                    robot.rotate(-rotation_step, world)
                if event.key == pygame.K_RIGHT:
                    robot.rotate(rotation_step, world)
                if event.key == pygame.K_UP:
                    robot.move(moving_step, world)
                if event.key == pygame.K_DOWN:
                    robot.move(-moving_step, world)

                sensors_view.take_measurements(odometry, sensor)
                slam_front_end.take_measurements(odometry, sensor)

        world.draw(screen)
        robot.draw(screen, world.height, world.width)
        if simulation_mode == SimulationMode.RAW_SENSORS:
            sensors_view.draw(screen, offset=world.width)
        if simulation_mode == SimulationMode.ICP_ADJUSTMENT:
            slam_front_end.draw(screen, offset=world.width)
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
