from pet_breathy.point import Point

import numpy as np

class PrngPointGen:
    def __init__(self, min_width, max_width, min_height, max_height):
        self.prng = np.random.default_rng(seed=1)
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
    
    def generate(self, num_points: int) -> list[Point]:
        x_points = self.prng.uniform(low=self.min_width, high=self.max_width, size=num_points)
        y_points = self.prng.uniform(low=self.min_height, high=self.max_height, size=num_points)
        points = [ ]
        #print('[')
        for x, y in zip(x_points, y_points):
            #print('    Point(%s, %s),' % (x, y))
            points.append(Point(x, y))
        #print(']')
        return points

    def generate_point(self) -> Point:
        x = self.prng.uniform(low=self.min_width, high=self.max_width)
        y = self.prng.uniform(low=self.min_height, high=self.max_height)
        return Point(x, y)
        
