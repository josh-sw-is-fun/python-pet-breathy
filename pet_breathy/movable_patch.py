from pet_breathy.patch import Patch
from pet_breathy.point import Point
from pet_breathy import signal_score
from pet_breathy.patch_type import PatchType

class MovablePatch(Patch):
    def __init__(self, patch_id: int, patch_center: Point, max_seg_len: int):
        super().__init__(patch_id, patch_center, max_seg_len, PatchType.MOVABLE)

def compare_based_on_signal_score(p0: MovablePatch, p1: MovablePatch) -> int:
    return signal_score.compare(p0.get_score(), p1.get_score())
