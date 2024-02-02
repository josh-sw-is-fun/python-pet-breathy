import cv2 as cv

class OpticalFlow:
    def __init__(self):
        # - (3,3)       Points are on bath salts
        # - (9,9)       Points go all over the place
        # - (21,21)     Points little eratic
        # - (51,51)     Seems ... to work alright
        # - (101,101)   Points do not move enough
        # - (201, 201)  Seem to be working alright for FFT analysis at least
        val = 71 #201
        self.optical_flow_params_win_size = (val, val)
        
        self.optical_flow_params_max_level = 3
        self.optical_flow_params_criteria = \
            (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
    
    def calc(self, prev_frame, curr_frame, prev_points):
        # st - status
        #   - Value of 1, feature has been found, otherwise 0
        # err - error
        #   - Error for the corresponding feature if found
        next_points, st, err = cv.calcOpticalFlowPyrLK(
            prev_frame,
            curr_frame,
            prev_points,
            None,
            winSize=self.optical_flow_params_win_size,
            maxLevel=self.optical_flow_params_max_level,
            criteria=self.optical_flow_params_criteria)
        
        return next_points, st, err
