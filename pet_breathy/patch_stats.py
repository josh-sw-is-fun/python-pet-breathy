from pet_breathy.patch_type import PatchType
from pet_breathy.patch_segment import PatchSegment
from pet_breathy.point import Point
from pet_breathy.signal import Signal
from pet_breathy.signal_score import SignalScore

import enum

class AnalysisState(enum.Enum):
    ANALYZING = enum.auto()
    # Patch was reset and reassigned to another location
    REASSIGNED = enum.auto()
    # Reset to its original location or some pseudo random location
    RESET = enum.auto()

class PatchFrameStats:
    def __init__(self, center: Point, frame_num: int, patch_seg: PatchSegment):
        self.point = center.clone()
        self.frame_num = frame_num
        self.state = None
        self.seg_length = None
        self.sig = None
        self.sig_score = None
        
        self.extended_info = patch_seg is not None
        self.y_avg_pts = None
        self.y_act_pts = None
        self.spectra = None
        
        if self.extended_info:
            seg = patch_seg.get_segment()
            overlapper = patch_seg.get_overlapper()
            
            # y_pts np.ndarray
            self.y_avg_pts = seg.get_avg_y_coords().copy()
            self.y_act_pts = seg.get_act_y_coords().copy()
            
            # spectra np.ndarray
            if overlapper.get_spectras_count():
                self.spectra = overlapper.get_spectra().copy()

    def set_stats(self, seg_length: int, sig: Signal, sig_score: SignalScore):
        self.seg_length = seg_length
        self.sig = sig
        self.sig_score = sig_score

    def set_state(self, state: AnalysisState):
        self.state = state

class PatchStats:
    def __init__(self, patch_id: int, patch_type: PatchType):
        self.patch_id = patch_id
        self.patch_type = patch_type
        self.frames = [ ]

    def add_frame(self, frame: PatchFrameStats):
        self.frames.append(frame)

    def set_frame_state(self, state: AnalysisState):
        self.frames[-1].set_state(state)
