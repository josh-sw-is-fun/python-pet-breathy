from pet_breathy.point_group_manager import PointGroupManager
from pet_breathy.point import Point

import unittest
import numpy as np

class TestPointGroupManager(unittest.TestCase):
    def test_empty(self):
        capacity = 2
        pgm = PointGroupManager(capacity)
        
        self.assertEqual(capacity, pgm.get_capacity())
        self.assertEqual(0, pgm.get_size())
        self.assertEqual(0, len(pgm.get_points()))

    def test_add_point_group(self):
        pgm = PointGroupManager(5)
        
        points = [
            Point(1,1),
            Point(2,2),
            Point(3,3),
            ]
        
        pg = pgm.create_point_group(0, points)

        self.assertEqual(len(points), pgm.get_size())
        self.assertEqual(5, pgm.get_capacity())

        pgm_points = pgm.get_points()
        self.assertEqual(len(points), len(pgm_points))
        
        self.assertTrue(np.all(pgm_points[0] == (1, 1)))
        self.assertTrue(np.all(pgm_points[1] == (2, 2)))
        self.assertTrue(np.all(pgm_points[2] == (3, 3)))
        
        self.assertEqual(0, pg.get_point_offset_idx())

    def test_add_point_group_multi(self):
        pgm = PointGroupManager(5)
        
        p0 = [ Point(1,1), Point(2,2), Point(3,3) ]
        p1 = [ Point(4,4), Point(5,5) ]

        pg0 = pgm.create_point_group(0, p0)
        pg1 = pgm.create_point_group(1, p1)
        
        pgm_points = pgm.get_points()
        
        self.assertEqual(len(pgm_points), pgm.get_size())
        self.assertEqual(len(p0) + len(p1), pgm.get_size())
        self.assertEqual(5, pgm.get_capacity())
        
        self.assertTrue(np.all(pgm_points[0] == (1, 1)))
        self.assertTrue(np.all(pgm_points[1] == (2, 2)))
        self.assertTrue(np.all(pgm_points[2] == (3, 3)))
        self.assertTrue(np.all(pgm_points[3] == (4, 4)))
        self.assertTrue(np.all(pgm_points[4] == (5, 5)))

        pg0_points = pg0.get_points()
        self.assertTrue(np.all(pg0_points[0] == (1, 1)))
        self.assertTrue(np.all(pg0_points[1] == (2, 2)))
        self.assertTrue(np.all(pg0_points[2] == (3, 3)))

        pg1_points = pg1.get_points()
        self.assertTrue(np.all(pg1_points[0] == (4, 4)))
        self.assertTrue(np.all(pg1_points[1] == (5, 5)))

