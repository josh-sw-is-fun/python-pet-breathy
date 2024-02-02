from pet_breathy.segment import Segment
from pet_breathy.fft_overlapper import FftOverlapper

class PatchSegment:
    def __init__(self, max_len: int, avg_kernel_size: int, num_overlaps: int):
        self.max_len = max_len
        self.avg_kernel_size = avg_kernel_size
        self.num_overlaps = num_overlaps

        self.segment = Segment(self.max_len, self.avg_kernel_size)
        self.overlapper = FftOverlapper(self.num_overlaps)

    def get_segment(self) -> Segment:
        return self.segment

    def get_overlapper(self) -> FftOverlapper:
        return self.overlapper

    def resize(self, new_size):
        self.overlapper = FftOverlapper(self.num_overlaps)
        
        new_segment = Segment(self.max_len, self.avg_kernel_size)
        
        y_vals = self.segment.get_latest_avg_y_coords(new_size)
        for y_val in y_vals:
            new_segment.append_y(y_val)
        self.segment = new_segment
