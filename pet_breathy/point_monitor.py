import numpy as np

# numpy ndarray indexing references:
# - https://note.nkmk.me/en/python-numpy-ndarray-compare/

class PointMonitor:
    # Too small of a jump, too many points are filtered
    # Too large of a jump messes with the frequency domain analysis (FFT of
    # time domain samples)
    MAX_JUMP_DIST = 50
    
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.out_of_frame_idxs = [ ]
        self.jump_idxs = [ ]

    def check_for_bad_points(self, prev_points: np.ndarray, curr_points: np.ndarray):
        '''
        @param prev_points Expect shape to be (N, 2) N elements
        @param curr_points Expect shape to be (N, 2)
        '''
        # A point is out of of frame if
        # - It is equal to or less than zero
        # - It is equal to or greater than the frame width/height
        self.out_of_frame_idxs, = np.where(
            np.any((curr_points <= (0, 0)) | (curr_points >= (self.frame_width, self.frame_height)), axis=1))
        # A point jumps if the distance between previous and current point
        # exceeds some threshold.
        self.jump_idxs, = np.where(
                # Calculate point distance, any point greater than some value
                # has "jumped". When points appear to jump like this, it really
                # screws up the FFT calculations when looking for a signal.
                np.linalg.norm(curr_points - prev_points, axis=1) > PointMonitor.MAX_JUMP_DIST)
    
    def get_out_of_frame_idxs(self):
        return self.out_of_frame_idxs
    
    def get_jump_idxs(self):
        return self.jump_idxs

    def has_bad_points(self):
        return len(self.out_of_frame_idxs) > 0 or len(self.jump_idxs) > 0
