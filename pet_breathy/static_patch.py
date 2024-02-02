from pet_breathy.patch import Patch
from pet_breathy.point_group_manager import PointGroupManager
from pet_breathy.point_group import PointGroup
from pet_breathy.patch_segment import PatchSegment
from pet_breathy.point import Point
from pet_breathy.patch_type import PatchType

class StaticPatch(Patch):
    def __init__(self,
            patch_id: int,
            patch_center: Point,
            max_seg_len: int,
            avg_kernel_size: int,
            num_overlaps: int,
            manager: PointGroupManager):
        
        super().__init__(patch_id, patch_center, max_seg_len, PatchType.STATIC)
        
        self.patch_seg = PatchSegment(
            self.max_seg_len, avg_kernel_size, num_overlaps)
        self.patch_seg.get_segment().append_y(self.patch_center.y)
        self.point_group = manager.create_point_group(self.get_id(), [self.patch_center])

    def reset(self):
        self.patch_seg.get_segment().reset()
        self.curr_center.copy(self.patch_center)
        self.patch_seg.get_segment().append_y(self.patch_center.y)
        self.point_group.reset([self.patch_center])

    def reset_center_point(self, new_center_point: Point):
        self.patch_center.copy(new_center_point)
        self.reset()

    def get_point_group(self) -> PointGroup:
        return self.point_group

    def points_updated(self):
        points = self.point_group.get_points()
        self.curr_center.x = points[0][0]
        self.curr_center.y = points[0][1]
        self.patch_seg.get_segment().append_y(points[0][1])

    def get_point_count(self) -> int:
        return self.patch_seg.get_segment().get_y_count()

    def get_largest_pow_of_2_point_count(self) -> int:
        return self.patch_seg.get_segment().get_largest_pow_of_2_y_point_count()

    def get_patch_segs(self) -> list:
        return [ self.patch_seg ]

    def get_patch_seg(self) -> PatchSegment:
        return self.patch_seg

    def set_debug_prints(self, enabled: bool):
        pass
