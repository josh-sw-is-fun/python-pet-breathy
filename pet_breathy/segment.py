from pet_breathy.point import Point
from pet_breathy.fixed_data_structures import FixedAvgQueue
from pet_breathy.point import Point
import pet_breathy.utils as utils

import numpy as np

class Segment:
    def __init__(self, max_len: int, avg_kernel_size: int):
        #self.x = FixedAvgQueue(max_len, float, avg_kernel_size)
        self.y = FixedAvgQueue(max_len, float, avg_kernel_size)
        self.largest_pow_of_2 = 0

    def append(self, p: Point):
        self.append_y(p.y)

    def append_y(self, y: float):
        self.y.append(y)
        y_size = self.y.get_size()
        if utils.is_pow_of_2(y_size):
            self.largest_pow_of_2 = y_size
    
    def get_avg_y_coords(self) -> np.ndarray:
        return self.y.get_avg_buf()
    
    def get_latest_avg_y_coords(self, count: int) -> np.ndarray:
        return self.y.get_latest_avg_buf(count)
    
    def get_act_y_coords(self) -> np.ndarray:
        return self.y.get_act_buf()
    
    def get_latest_act_y_coords(self, count: int) -> np.ndarray:
        return self.y.get_latest_act_buf(count)
    
    def get_y_count(self) -> int:
        return self.y.get_size()
    
    def get_largest_pow_of_2_y_point_count(self) -> int:
        return self.largest_pow_of_2
    
    def reset(self):
        self.y.clear_no_erase()
        self.largest_pow_of_2 = 0
