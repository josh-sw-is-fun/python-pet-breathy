from pet_breathy.point_monitor import PointMonitor

import unittest
import numpy as np

class TestPointMonitor(unittest.TestCase):
    def test_all_good_points(self):
        mon = PointMonitor(10, 10)
        
        prev = np.ndarray((2, 2), dtype=int)
        curr = np.ndarray((2, 2), dtype=int)
        
        prev[0] = (1, 1)
        prev[1] = (2, 2)
        curr[0] = (2, 2)
        curr[1] = (3, 3)
        
        mon.check_for_bad_points(prev, curr)
        
        self.assertFalse(mon.has_bad_points())
    
    def test_point_jump(self):
        mon = PointMonitor(5, 5)
        
        prev = np.ndarray((3, 2), dtype=int)
        curr = np.ndarray((3, 2), dtype=int)
        
        prev[0] = (3, 3)
        prev[1] = (1, 1)    # < index 1
        prev[2] = (2, 2)
        curr[0] = (2, 2)
        curr[1] = (PointMonitor.MAX_JUMP_DIST + 1, PointMonitor.MAX_JUMP_DIST + 1)    # < index 1, jump
        curr[2] = (1, 1)
        
        mon.check_for_bad_points(prev, curr)
        
        self.assertTrue(mon.has_bad_points())
        
        jumped = mon.get_jump_idxs()
        out_of_frame = mon.get_out_of_frame_idxs()
        
        self.assertEqual(1, len(out_of_frame))
        self.assertEqual(1, out_of_frame[0])
        
        self.assertEqual(1, len(jumped))
        # Expected the jump index to be at index 1
        self.assertEqual(1, jumped[0])

    def test_out_of_frame(self):
        dim = 10
        mon = PointMonitor(dim, dim)
        
        prev = np.ndarray((3, 2), dtype=int)
        curr = np.ndarray((3, 2), dtype=int)
        
        # Prev points are not checked for being out of frame, just current points
        prev[0] = (1, 2)
        prev[1] = (9, 9)
        prev[2] = (1, 9)
        curr[0] = (0, 2)    # <- out of frame
        curr[1] = (9, 9)
        curr[2] = (1, dim)  # <- out of frame
        
        mon.check_for_bad_points(prev, curr)
        
        self.assertTrue(mon.has_bad_points())
        
        jumped = mon.get_jump_idxs()
        out_of_frame = mon.get_out_of_frame_idxs()
        
        # check that there are not any jumped points
        self.assertEqual(0, len(jumped))
        
        self.assertEqual(2, len(out_of_frame))

        self.assertEqual(0, out_of_frame[0])
        self.assertEqual(2, out_of_frame[1])

    def test_jump_and_out_of_frame(self):
        dim = PointMonitor.MAX_JUMP_DIST + 1
        mon = PointMonitor(dim, dim)
        
        prev = np.ndarray((1, 2), dtype=int)
        curr = np.ndarray((1, 2), dtype=int)
        
        # Prev points are not checked for being out of frame, just current points
        prev[0] = (0, 0)
        curr[0] = (dim, dim)  # <- out of frame and jump
        
        mon.check_for_bad_points(prev, curr)
        
        self.assertTrue(mon.has_bad_points())
        
        jumped = mon.get_jump_idxs()
        out_of_frame = mon.get_out_of_frame_idxs()
        
        self.assertEqual(0, jumped[0])
        self.assertEqual(0, out_of_frame[0])
