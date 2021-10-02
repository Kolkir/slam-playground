class Frame:
    def __init__(self, position, rotation, observed_points):
        self.transform = position
        self.rotation = rotation
        self.__observed_points = observed_points

    @property
    def observed_points(self):
        return self.__observed_points
