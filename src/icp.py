from sklearn.neighbors import NearestNeighbors


class ICP:
    """
        Iterative Closest Point (ICP) implementation
    """
    def __init__(self, num_neighbors=1):
        self.__knn = NearestNeighbors(num_neighbors)

    def __nearest_neighbors(self, points_a, points_b):
        self.__knn.fit(points_b)
        distances, indices = self.__knn.kneighbors(points_a, return_distance=True)
        return distances.ravel(), indices.ravel()

    def find_transform(self, points_a, points_b):
        neighbors, indices = self.__nearest_neighbors(points_a, points_b)