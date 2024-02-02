from pet_breathy.optical_flow import OpticalFlow

import numpy as np
import datetime as dt

class OpticalFlowPointBudgetCalculator:
    def __init__(self, m, b):
        '''
        @param m The slope of the interpolated point budget measurement
        @param b Y-intercept of the interpolated point budget measurement
        '''
        self.m = m
        self.b = b
    
    def calc_points(self, secs: float) -> float:
        '''Return the number of points that can be used given the time
        @param secs Interval to calculate the number of points that can be used
               in optical flow calculation
        @return num_points Number of points that could be used given the input
                time
        '''
        # y = mx + b -> x = (y - b) / m
        return (secs - self.b) / self.m

def create(prev_frame: np.ndarray, curr_frame: np.ndarray) -> OpticalFlowPointBudgetCalculator:
    ''' Calculates the runtime to run an optical flow calculation with A
    points then the same thing with B points. Based on the runtime of both
    calculations, linear interpolate to find the number of points or point
    budget that can be used to find signals.
    
    For example, if we wish to take .2 seconds per frame after decimation,
    then we could calculate the number of points that can be used while
    keeping the runtime around 75% (for example) of .2 seconds.
    
    This is just for a rough estimate of run time not exact. I would rather
    measure the number of points I can use then guess at 300 or something
    like that.
    
    The current and previous frames are fed into the optical flow object.
    The runtime of this operation is measured.
    '''
    a_point_count = 100
    b_point_count = 1000
    
    a_runtime = _measure_runtime(a_point_count, prev_frame, curr_frame)
    b_runtime = _measure_runtime(b_point_count, prev_frame, curr_frame)
    
    x0 = a_point_count
    y0 = a_runtime
    
    x1 = b_point_count
    y1 = b_runtime
    
    m = (y1 - y0) / (x1 - x0)
    
    # y = mx + b -> b = y - mx
    b = y0 - m * x0
    
    return OpticalFlowPointBudgetCalculator(m, b)

def _measure_runtime(
        point_count: int,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray) -> float:
    
    flow = OpticalFlow()
    frame_height, frame_width, frame_depth = prev_frame.shape
    
    points = _create_points(point_count, frame_height, frame_width)
    
    t0 = dt.datetime.now()
    flow.calc(prev_frame, curr_frame, points)
    t1 = dt.datetime.now()
    
    elapsed = (t1 - t0).total_seconds()
    
    print('Point count: %s, Elapsed: %s' % (point_count, elapsed))
    
    return elapsed

def _create_points(point_count: int, frame_height: int, frame_width: int) -> np.ndarray:
    if point_count > (frame_height * frame_width):
        raise Exception('Cannot create points for optical flow point budget '
            'calculator, expecting frame dimiensions to be bigger')
    
    points = np.ndarray((point_count, 2), dtype=np.float32)
    for i in range(point_count):
        x = i % frame_width
        y = (i // frame_height) * frame_width
        points[i] = (x, y)
    return points

