from pet_breathy.point import Point
from pet_breathy.segment import Segment


class FrameSegment:
    def __init__(self,
            starting_point: Point,
            frame_width: int,
            frame_height: int,
            max_seg_point_count: int):
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        if not self._point_in_frame(starting_point):
            raise Exception('Starting point not in frame')
        self.in_frame = True
        
        self.seg = Segment(max_seg_point_count)
        self.seg.append(starting_point)

    def add_point(self, p: Point):
        if self._point_in_frame(p):
            self.seg.append(p)
        else:
            self._went_out_of_frame()

    def is_in_frame(self): -> bool
        return self.in_frame

    def _went_out_of_frame(self):
        self.in_frame = False

    def _point_in_frame(self, p: Point): -> bool
        return 0 < p.x < self.frame_width and 0 < p.y < self.frame_height


class DoOverFrameSegment(FrameSegment):
    def __init__(self,
            starting_point: Point,
            frame_width: int,
            frame_height: int,
            max_seg_point_count: int):
        
        super().__init__(
            starting_point,
            frame_width,
            frame_height,
            max_seg_point_count)
        
        self.starting_point = starting_point
    
    def _went_out_of_frame(self):
        self.seg.reset()
        self.seg.append(self.starting_point)
