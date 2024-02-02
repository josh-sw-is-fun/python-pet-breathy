from pet_breathy.movable_patch import MovablePatch
from pet_breathy.point_group_manager import PointGroupManager
from pet_breathy.point_group import PointGroup
from pet_breathy.point import Point
from pet_breathy.patch_segment import PatchSegment
from pet_breathy.signal import Signal
from pet_breathy.signal_info import SignalInfo
from pet_breathy.signal_score import SignalScore
from pet_breathy.signal_analyzer_result import SignalAnalyzerResult
from pet_breathy.point_signal import PointSignal
from pet_breathy import signal_analyzer
from pet_breathy.patch_state_machine import PatchStateMachine
from pet_breathy.patch_stats import AnalysisState, PatchFrameStats

import numpy as np
import scipy as sp


class CirclePatch(MovablePatch):
    NUM_POINTS = 9
    def __init__(self,
            patch_id: int,
            patch_center: Point,
            max_seg_len: int,
            avg_kernel_size: int,
            num_overlaps: int,
            manager: PointGroupManager,
            point_dist: float,
            sig_info: SignalInfo):
        
        super().__init__(patch_id, patch_center, max_seg_len)
        
        self.avg_kernel_size = avg_kernel_size
        self.num_overlaps = num_overlaps
        self.point_dist = point_dist
        self.sig_info = sig_info
        
        self.num_patch_segs = CirclePatch.NUM_POINTS
        self.patch_segs = [ ]
        
        self.points = [ ]
        
        for i in range(self.num_patch_segs):
            self.patch_segs.append(
                PatchSegment(self.max_seg_len, self.avg_kernel_size, self.num_overlaps))
            self.points.append(Point(0, 0))
        
        # Start center represents the starting location of the center point before it reset
        self.start_center = self.patch_center.clone()
        
        # Pythagorean theorum: A**2 + B**2 = C**2, I want A and B to be equal:
        #
        # A**2 + A**2       -> 2 * A**2
        # 2 * A**2 = C**2   -> A = sqrt(C**2 / 2)
        #
        self.a_dist = round(np.sqrt((self.point_dist * self.point_dist) / 2))
        
        self._update_points()
        
        for patch_seg, point in zip(self.patch_segs, self.points):
            patch_seg.get_segment().append_y(point.y)
        
        self.point_group = manager.create_point_group(self.get_id(), self.points)
        self.sm = PatchStateMachine(self.get_id(), self.patch_segs, self.sig_info)
        
        # The PointSignal for the strongest signal for this patch. If the
        # strongest signal is the center point, then this is set to None to
        # indicate there isn't a stronger point.
        self.best_point_signal = None
        
        self.use_new_point_on_reset = True

    def reset(self):
        self.start_center.copy(self.patch_center)
        self.curr_center.copy(self.patch_center)
        self._update_points()
        
        for patch_seg, point in zip(self.patch_segs, self.points):
            patch_seg.get_segment().reset()
            patch_seg.get_segment().append_y(point.y)
        
        self.point_group.reset(self.points)
        
        self.sm.reset()

    def reset_center_point(self, new_center_point: Point):
        if self.use_new_point_on_reset:
            self.patch_center.copy(new_center_point)
        self.reset()

    def get_point_group(self) -> PointGroup:
        return self.point_group
    
    def points_updated(self):
        points = self.point_group.get_points()
        self.curr_center.x = points[0][0]
        self.curr_center.y = points[0][1]
        for patch_seg, point in zip(self.patch_segs, points):
            # - point[0] = x coordinate
            # - point[1] = y coordinate
            patch_seg.get_segment().append_y(point[1])

    def get_point_count(self) -> int:
        return self.patch_segs[0].get_segment().get_y_count()

    def get_largest_pow_of_2_point_count(self) -> int:
        return self.patch_segs[0].get_segment().get_largest_pow_of_2_y_point_count()

    def get_patch_segs(self) -> list:
        return self.patch_segs

    # ^^^ Patch overrides ^^^
    ###########################################################################
    
    def add_frame_stats(self, stats: PatchFrameStats):
        self.get_stats().add_frame(stats)
    
    def frame_stats_set_state(self, state: AnalysisState):
        self.get_stats().set_frame_state(state)
    
    def analyze_new_signals(self, results: list[SignalAnalyzerResult]):
        self.sm.process(results[0].signals, results[0].yf_points)
        
        self.best_point_signal = None
        if self.sm.has_good_freq_time_domain():
            best_strength = 0.0
            best_signal = None
            best_point = None
            for i in range(1, len(results)):
                signals = results[i].signals
                for signal in signals:
                    if signal.strength > best_strength:
                        best_strength = signal.strength
                        best_signal = signal
                        best_point = self.points[i]
            
            self.best_point_signal = PointSignal(best_point, best_signal)
    
    def failed(self):
        return self.sm.failed()
    
    def get_score(self) -> SignalScore:
        return self.sm.get_score()
    
    def get_best_point_signal(self) -> PointSignal:
        '''Next best signal that could be used to relocate a patch to.'''
        return self.best_point_signal
    
    def do_not_use_new_point_on_reset(self):
        '''This is for debug purposes.'''
        self.use_new_point_on_reset = False
    
    def get_debug_signal(self):
        return self.sm.get_debug_signal()
    
    def set_debug_prints(self, enabled: bool):
        self.debug_prints = enabled
        self.sm.set_debug_prints(enabled)
    
    def _update_points(self):
        '''
        Patch of 9 points in a circle
        
            |---| --> a_dist
            x
        x       x --> 1:30, outer points are oriented like a clock
      
      x     x     x
      
        x       x --> 4:30
            x
            |-----| --> point_dist
        '''
        p = self.points
        c = self.start_center
        
        p[0].x, p[0].y = c.x,                   c.y                        # center
        p[1].x, p[1].y = c.x,                   c.y - self.point_dist      # 12:00 o'clock
        p[2].x, p[2].y = c.x + self.a_dist,     c.y - self.a_dist          #  1:30
        p[3].x, p[3].y = c.x + self.point_dist, c.y                        #  3:00
        p[4].x, p[4].y = c.x + self.a_dist,     c.y + self.a_dist          #  4:30
        p[5].x, p[5].y = c.x,                   c.y + self.point_dist      #  6:00
        p[6].x, p[6].y = c.x - self.a_dist,     c.y + self.a_dist          #  7:30
        p[7].x, p[7].y = c.x - self.point_dist, c.y                        #  9:00
        p[8].x, p[8].y = c.x - self.a_dist,     c.y - self.a_dist          # 11:00
