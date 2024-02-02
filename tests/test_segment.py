import unittest

from pet_breathy.segment import Segment
from pet_breathy.point import Point

class TestSegment(unittest.TestCase):
    def test_empty(self):
        seg = Segment(0, 3)
        self.assertEqual(0, len(seg.get_act_y_coords()))

    def test_append(self):
        seg = Segment(1, 3)
        seg.append(Point(123, 456))
        y_coord = seg.get_act_y_coords()
        self.assertEqual(1, len(y_coord))
        self.assertEqual(456, y_coord[0])
        
        seg.append(Point(654, 321))
        
        # Testing that y_coord still points to the array even after adding
        self.assertEqual(321, y_coord[0])
        
        # Get the reference anyway 
        y_coord = seg.get_act_y_coords()
        self.assertEqual(1, len(y_coord))

    def test_2_elements(self):
        ''' Testing the contents of array as elements are appended '''
        seg = Segment(2, 3)
        seg.append(Point(1, 2))
        y_coord = seg.get_act_y_coords()
        
        self.assertEqual(1, len(y_coord))
        self.assertEqual(2, y_coord[0])
        
        seg.append(Point(3, 4))
        
        self.assertEqual(1, len(y_coord))
        self.assertEqual(2, y_coord[0])
        
        y_coord = seg.get_act_y_coords()
        
        self.assertEqual(2, len(y_coord))
        self.assertEqual(2, y_coord[0])
        self.assertEqual(4, y_coord[1])
        
        seg.append(Point(5, 6))
        
        # Didn't call get_act_y_coords, checking the that the reference we already
        # have is still good
        self.assertEqual(2, len(y_coord))
        self.assertEqual(4, y_coord[0])
        self.assertEqual(6, y_coord[1])

    def test_get_latest_y_coords(self):
        seg = Segment(3, 3)
        seg.append(Point(1, 2))
        seg.append(Point(3, 4))
        seg.append(Point(5, 6))
        
        y_coord = seg.get_latest_act_y_coords(1)
        self.assertEqual(1, len(y_coord))
        self.assertEqual(6, y_coord[0])
        
        y_coord = seg.get_latest_act_y_coords(2)
        self.assertEqual(2, len(y_coord))
        self.assertEqual(4, y_coord[0])
        self.assertEqual(6, y_coord[1])
        
        y_coord = seg.get_latest_act_y_coords(3)
        self.assertEqual(3, len(y_coord))
        self.assertEqual(2, y_coord[0])
        self.assertEqual(4, y_coord[1])
        self.assertEqual(6, y_coord[2])

    def test_get_largest_pow_of_2_y_point_count(self):
        seg = Segment(10, 3)
        self.assertEqual(0, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(0, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(1, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(1, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(2, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(2, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(2, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(3, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(4, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(4, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(4, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(5, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(4, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(6, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(4, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(7, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(8, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(8, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(8, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(9, seg.get_y_count())
        
        seg.append(Point(0, 0))
        self.assertEqual(8, seg.get_largest_pow_of_2_y_point_count())
        self.assertEqual(10, seg.get_y_count())

if __name__ == '__main__':
    unittest.main()
