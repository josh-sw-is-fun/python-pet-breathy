import numpy as np

class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return '{ %s, %s }' % (self.x, self.y)

    def copy(self, other):
        self.x = other.x
        self.y = other.y

    def clone(self):
        return Point(self.x, self.y)

def calc_point_dist(p0: Point, p1: Point) -> float:
    return np.sqrt(((p1.x - p0.x)**2) + ((p1.y - p0.y)**2))
