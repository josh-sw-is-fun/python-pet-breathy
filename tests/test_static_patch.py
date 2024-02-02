from pet_breathy.static_patch import StaticPatch
from pet_breathy.point import Point
from pet_breathy.point_group_manager import PointGroupManager

import unittest
import numpy as np

class TestStaticPatch(unittest.TestCase):
    def test_basic(self):
        sp = self._make_basic_static_patch()
        
        self.assertEqual(1, sp.get_point_count())
        self.assertEqual(1, sp.get_largest_pow_of_2_point_count())
        
        pg = sp.get_point_group()

        self.assertEqual(sp.get_id(), pg.get_group_id())

        status = pg.get_status()
        
        self.assertFalse(status.did_point_go_out_of_frame())
        self.assertFalse(status.did_point_jump())

    def test_reset(self):
        sp = self._make_basic_static_patch()
        
        pg = sp.get_point_group()
        patch_segs = sp.get_patch_segs()
        
        points = pg.get_points()
        orig_points = points.copy()
        
        points += 10
        self.assertFalse(np.array_equal(orig_points, points))
        
        for patch_seg in patch_segs:
            seg = patch_seg.get_segment()
            self.assertEqual(1, seg.get_y_count())
            self.assertEqual(1, len(seg.get_act_y_coords()))
            self.assertEqual(1, len(seg.get_avg_y_coords()))
        
        sp.points_updated()
        
        for patch_seg in patch_segs:
            seg = patch_seg.get_segment()
            self.assertEqual(2, seg.get_y_count())
            self.assertEqual(2, len(seg.get_act_y_coords()))
            self.assertEqual(2, len(seg.get_avg_y_coords()))
        
        sp.reset()
        self.assertTrue(np.array_equal(orig_points, points))
        
        for patch_seg in patch_segs:
            seg = patch_seg.get_segment()
            self.assertEqual(1, seg.get_y_count())
            self.assertEqual(1, len(seg.get_act_y_coords()))
            self.assertEqual(1, len(seg.get_avg_y_coords()))

    def _make_basic_static_patch(self) -> StaticPatch:
        center_point = Point(100, 100)
        max_seg_len = 32
        avg_kernel_size = 7
        manager = PointGroupManager(100)
        
        sp = StaticPatch(
            0,
            center_point,
            max_seg_len,
            avg_kernel_size,
            30 // 6,
            manager)
        
        return sp
