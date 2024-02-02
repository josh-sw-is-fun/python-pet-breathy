from pet_breathy.point import Point
from pet_breathy.patch_type import PatchType
from pet_breathy.patch_stats import PatchStats

class Patch:
    def __init__(self,
            patch_id: int,
            patch_center: Point,
            max_seg_len: int,
            patch_type: PatchType):
        self.patch_id = patch_id
        self.patch_center = patch_center.clone()
        self.max_seg_len = max_seg_len
        self.patch_type = patch_type
        self.stats = PatchStats(patch_id, patch_type)
        
        # Current center represents the latest location of the center
        self.curr_center = self.patch_center.clone()

    def get_id(self) -> int:
        return self.patch_id

    def get_stats(self) -> PatchStats:
        return self.stats

    def get_center_point(self) -> Point:
        return self.curr_center
