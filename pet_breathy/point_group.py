from pet_breathy.point_group_status import PointGroupStatus
from pet_breathy.point import Point

import numpy as np

class PointGroup:
    def __init__(self, group_id: int, offset_idx: int, points: np.ndarray):
        self.group_id = group_id
        self.offset_idx = offset_idx
        self.points = points
        
        self.status = PointGroupStatus()

    def get_group_id(self) -> int:
        return self.group_id

    def get_point_offset_idx(self) -> int:
        return self.offset_idx

    def get_points(self) -> np.ndarray:
        return self.points

    def get_status(self) -> PointGroupStatus:
        return self.status

    def update_points(self, points: list[Point]):
        for i, p in enumerate(points):
            self.points[i] = (p.x, p.y)

    def reset(self, points: list[Point]):
        self.status.reset()
        self.update_points(points)
