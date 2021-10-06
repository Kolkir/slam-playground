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
    sensor = Sensor(dist_range=350, fov=90, mu=0, sigma=0)  # noised measurements
    robot = Robot(odometry, sensor)
    sensors_view = RawSensorsView(world.height, world.width)
    slam_front_end = playground.slam.frontend.FrontEnd(world.height, world.width)

    # Initialize rendering
    screen = pygame.display.set_mode([world.width * 2, world.height])
    font = pygame.font.Font(pygame.font.get_default_font(), 24)
    sensors_text_surface = font.render('Sensors', True, (255, 0, 0))
    icp_text_surface = font.render('ICP', True, (255, 0, 0))
    slam_text_surface = font.render('Pose Graph', True, (255, 0, 0))
    text_pos = (15, 15)

    # Robot movement configuration
    rotation_step = 10  # degrees
    moving_step = 10  # points

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
                    robot.rotate(rotation_step, world)
                if event.key == pygame.K_RIGHT:
                    robot.rotate(-rotation_step, world)
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
            screen.blit(sensors_text_surface, dest=text_pos)
        if simulation_mode == SimulationMode.ICP_ADJUSTMENT:
            slam_front_end.draw(screen, offset=world.width)
            screen.blit(icp_text_surface, dest=text_pos)
        if simulation_mode == SimulationMode.ICP_POSE_GRAPH_ADJUSTMEN:
            # TODO: Draw SLAM results
            screen.blit(slam_text_surface, dest=text_pos)


        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
