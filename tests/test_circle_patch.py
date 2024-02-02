from pet_breathy.circle_patch import CirclePatch
from pet_breathy.point import Point
from pet_breathy.point_group_manager import PointGroupManager
from pet_breathy.signal_info import SignalInfo

import unittest
import numpy as np

class TestCirclePatch(unittest.TestCase):
    def test_basic(self):
        cp = self._make_basic_circle_patch()
        
        self.assertEqual(1, cp.get_point_count())
        self.assertEqual(1, cp.get_largest_pow_of_2_point_count())
        
        pg = cp.get_point_group()

        self.assertEqual(cp.get_id(), pg.get_group_id())

        status = pg.get_status()
        
        self.assertFalse(status.did_point_go_out_of_frame())
        self.assertFalse(status.did_point_jump())

    def test_reset(self):
        cp = self._make_basic_circle_patch()
        
        pg = cp.get_point_group()
        patch_segs = cp.get_patch_segs()
        
        points = pg.get_points()
        orig_points = points.copy()
        
        points += 10
        self.assertFalse(np.array_equal(orig_points, points))
        
        for patch_seg in patch_segs:
            seg = patch_seg.get_segment()
            self.assertEqual(1, seg.get_y_count())
            self.assertEqual(1, len(seg.get_act_y_coords()))
            self.assertEqual(1, len(seg.get_avg_y_coords()))
        
        cp.points_updated()
        
        for patch_seg in patch_segs:
            seg = patch_seg.get_segment()
            self.assertEqual(2, seg.get_y_count())
            self.assertEqual(2, len(seg.get_act_y_coords()))
            self.assertEqual(2, len(seg.get_avg_y_coords()))
        
        cp.reset()
        self.assertTrue(np.array_equal(orig_points, points))
        
        for patch_seg in patch_segs:
            seg = patch_seg.get_segment()
            self.assertEqual(1, seg.get_y_count())
            self.assertEqual(1, len(seg.get_act_y_coords()))
            self.assertEqual(1, len(seg.get_avg_y_coords()))

    def _make_basic_circle_patch(self) -> CirclePatch:
        center_point = Point(100, 100)
        max_seg_len = 32
        avg_kernel_size = 7
        manager = PointGroupManager(100)
        point_dist = 15
        sig = SignalInfo()
        sig.decimation = 6
        sig.fps = 30
        
        cp = CirclePatch(
            0,
            center_point,
            max_seg_len,
            avg_kernel_size,
            30 // 6,
            manager,
            point_dist,
            sig)
        
        return cp
