class Frame:
    def __init__(self, transform, observed_points):
        self.transform = transform
        self.__observed_points = observed_points

    @property
    def observed_points(self):
        return self.__observed_points
