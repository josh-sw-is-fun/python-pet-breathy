from pet_breathy.point import Point
from pet_breathy.point_group import PointGroup
from pet_breathy.point_monitor import PointMonitor
from pet_breathy.fixed_data_structures import FixedArray

import numpy as np

class PointGroupManager:
    def __init__(self, max_points: int):
        self.max_points = max_points

        self.point_idx = 0
        
        # Key:      id: int
        # Value:    PointGroup
        self.groups = { }
        
        # Lookup to map point index to group id
        self.point_idx_to_group_id_lookup = np.ndarray(self.max_points, dtype=int)
        
        # The shape and data type is compatible with OpticalFlow input/output.
        # Example if max_points was 1 - [[ 36. 178.]], shape: (1, 2)
        self.points = FixedArray((self.max_points, 2), np.float32)
        
        # List of group ids that have bad points
        self.bad_group_ids = [ ]
        
        self.debug = False

    def get_capacity(self) -> int:
        return self.max_points

    def get_size(self) -> int:
        return self.point_idx

    def get_points(self) -> np.ndarray:
        return self.points.get_buf()

    def get_point_group(self, group_id: int) -> PointGroup:
        return self.groups[group_id]

    def create_point_group(self, group_id: int, points: list[Point]) -> PointGroup:
        if self.point_idx + len(points) > self.max_points:
            raise Exception('Not enough room to add points')
        
        start_idx = self.point_idx
        
        for point in points:
            self.points.append((point.x, point.y))
            
            self.point_idx_to_group_id_lookup[self.point_idx] = group_id
            
            self.point_idx += 1
        
        point_buf = self.points.get_buf()
        
        point_group = PointGroup(
            group_id,
            start_idx,
            point_buf[start_idx:self.point_idx])
        
        if group_id in self.groups:
            raise RuntimeError('Cannot create point group, id already in use')
        self.groups[group_id] = point_group
        
        return point_group

    def update_points(self, new_points: np.ndarray, point_monitor: PointMonitor):
        out_of_frame_gids = set()
        jump_gids = set()
        
        if point_monitor.has_bad_points():
            out_of_frame_idxs = point_monitor.get_out_of_frame_idxs()
            
            for idx in out_of_frame_idxs:
                group_id = self.point_idx_to_group_id_lookup[idx]
                self.groups[group_id].get_status().point_went_out_of_frame()
                out_of_frame_gids.add(int(group_id))
            
            jump_idxs = point_monitor.get_jump_idxs()
            
            for idx in jump_idxs:
                group_id = self.point_idx_to_group_id_lookup[idx]
                self.groups[group_id].get_status().point_jumped()
                jump_gids.add(int(group_id))
            
            if self.debug:
                print('out: %s, jump: %s' % (out_of_frame_gids, jump_gids))
        
        self.bad_group_ids = out_of_frame_gids | jump_gids
        self.points.copy(new_points)

    def get_bad_group_ids(self) -> list:
        return self.bad_group_ids

    def has_bad_group_ids(self) -> bool:
        return len(self.bad_group_ids) > 0
