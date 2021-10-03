import numpy as np

from playground.environment.body import Body


class Odometry(Body):
    def __init__(self, mu, sigma):
        super().__init__()
        self.__mu = mu
        self.__sigma = sigma

    def track_rotate(self, angle):
        noise = np.random.normal(self.__mu, self.__sigma)
        self.rotate(angle + noise)

    def track_move(self, dist):
        noise = np.random.normal(self.__mu, self.__sigma)
        self.move(dist + noise)
