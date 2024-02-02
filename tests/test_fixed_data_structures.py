from pet_breathy.fixed_data_structures import (
    FixedBuffer,
    FixedArray,
    FixedQueue,
    FixedAvgQueue)

import unittest
import numpy as np


class TestFixedBuffer(unittest.TestCase):
    def test_empty(self):
        capacity = 2
        buf = FixedBuffer(capacity, int)
        self.assertEqual(0, buf.get_size())
        self.assertEqual(capacity, buf.get_capacity())


class TestFixedArray(unittest.TestCase):
    def test_append_not_full(self):
        ''' Append to an array but do not hit capacity '''
        capacity = 2
        arr = FixedArray(capacity, int)
        
        arr.append(9)
        self.assertEqual(1, arr.get_size())
        self.assertEqual(capacity, arr.get_capacity())
        
        buf = arr.get_buf()
        self.assertEqual(9, buf[0])
        self.assertEqual(1, len(buf))

    def test_append_to_full(self):
        ''' Fill array '''
        capacity = 2
        arr = FixedArray(capacity, int)
        
        arr.append(1)
        arr.append(2)
        self.assertEqual(capacity, arr.get_size())
        self.assertEqual(capacity, arr.get_capacity())
        
        buf = arr.get_buf()
        self.assertEqual(1, buf[0])
        self.assertEqual(2, buf[1])

    def test_updates_persist_after_append(self):
        ''' Add element, get buffer, update element, then append again, make
        sure all elements are the value they should be
        '''
        capacity = 2
        arr = FixedArray(capacity, int)
        
        arr.append(9)
        
        buf = arr.get_buf()
        self.assertEqual(9, buf[0])
        
        # Update the array with the buffer we got back, want to ensure the
        # ndarray slice we get back is a reference and not a copy
        buf[0] = 123
        self.assertEqual(123, buf[0])
        
        arr.append(10)
        
        # Just updated array, ensure buffer slice hasn't been modified
        self.assertEqual(1, len(buf))
        
        buf = arr.get_buf()
        self.assertEqual(2, len(buf))
        
        self.assertEqual(buf[0], 123)
        self.assertEqual(buf[1], 10)
    
    def test_exceed_size(self):
        ''' Add too many elements '''
        arr = FixedArray(1, int)
        arr.append(1)
        
        with self.assertRaises(Exception):
            arr.append(2)

        buf = arr.get_buf()
        
        # After exception was raised, check that the object is still usable
        self.assertEqual(1, buf[0])
        self.assertEqual(1, arr.get_size())
        self.assertEqual(1, arr.get_capacity())

    def test_clear_no_erase(self):
        arr = FixedArray(1, int)
        arr.append(1)
        
        self.assertEqual(1, arr.get_size())
        
        arr.clear_no_erase()
        
        self.assertEqual(0, arr.get_size())
        arr.append(123)
        self.assertEqual(1, arr.get_size())
        
        buf = arr.get_buf()
        self.assertEqual(123, buf[0])


class TestFixedQueue(unittest.TestCase):
    def test_append(self):
        q = FixedQueue(1, int)
        q.append(5)
        self.assertEqual(1, q.get_size())
        self.assertEqual(1, q.get_capacity())
        
        buf = q.get_buf()
        self.assertEqual(5, buf[0])
        
        q.append(10)
        
        # Checking that the array slice is updated too
        self.assertEqual(10, buf[0])
        
        self.assertEqual(1, q.get_size())
        self.assertEqual(1, q.get_capacity())
        
        # Checking new buf slice is the same
        buf = q.get_buf()
        self.assertEqual(10, buf[0])

    def test_get_latest_buf(self):
        q = FixedQueue(5, int)
        
        q.append(1)
        q.append(2)
        q.append(3)
        
        buf = q.get_latest_buf(1)
        self.assertEqual(1, len(buf))
        self.assertEqual(buf[0], 3)
        
        buf = q.get_latest_buf(2)
        self.assertEqual(2, len(buf))
        self.assertEqual(buf[0], 2)
        self.assertEqual(buf[1], 3)
        
        buf = q.get_latest_buf(3)
        self.assertEqual(3, len(buf))
        self.assertEqual(buf[0], 1)
        self.assertEqual(buf[1], 2)
        self.assertEqual(buf[2], 3)
        
        q.append(4)
        
        buf = q.get_latest_buf(1)
        self.assertEqual(1, len(buf))
        self.assertEqual(4, buf[0])


class TestFixedAvgQueue(unittest.TestCase):
    def test_simple_append(self):
        q = FixedAvgQueue(1, int, 5)
        q.append(1)
        b = q.get_buf()
        self.assertEqual(1, b[0])

    def test_append_2_elements(self):   
        q = FixedAvgQueue(2, float, 5)
        vals = [1, 2]
        for val in vals:
            q.append(val)
        b = q.get_avg_buf()
        self.assertEqual(np.average(vals), b[0])
        self.assertEqual(np.average(vals), b[1])

    def test_append_more_than_kernel_size_worth(self):
        kernel_size = 5
        q = FixedAvgQueue(9, float, kernel_size)
        
        #       0  1  2  3  4  5  6   7  8  9  10   - 11 elements
        vals = [6, 4, 3, 1, 9, 9, 6, 10, 5, 4, 10]

        q.append(vals[0])
        self.assertEqual(6, q.get_buf()[0])
        
        q.append(vals[1])
        self.assertEqual(np.average(vals[0:2]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[0:2]), q.get_avg_buf()[1])
        
        q.append(vals[2])
        self.assertEqual(np.average(vals[0:3]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[0:3]), q.get_avg_buf()[1])
        self.assertEqual(np.average(vals[0:3]), q.get_avg_buf()[2])
        
        q.append(vals[3])
        self.assertEqual(np.average(vals[0:3]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[0:4]), q.get_avg_buf()[1])
        self.assertEqual(np.average(vals[0:4]), q.get_avg_buf()[2])
        self.assertEqual(np.average(vals[1:4]), q.get_avg_buf()[3])
        
        q.append(vals[4])
        self.assertEqual(np.average(vals[0:3]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[0:4]), q.get_avg_buf()[1])
        self.assertEqual(np.average(vals[0:5]), q.get_avg_buf()[2])
        self.assertEqual(np.average(vals[1:5]), q.get_avg_buf()[3])
        self.assertEqual(np.average(vals[2:5]), q.get_avg_buf()[4])
        
        q.append(vals[5])
        self.assertEqual(np.average(vals[0:3]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[0:4]), q.get_avg_buf()[1])
        self.assertEqual(np.average(vals[0:5]), q.get_avg_buf()[2])
        self.assertEqual(np.average(vals[1:6]), q.get_avg_buf()[3])
        self.assertEqual(np.average(vals[2:6]), q.get_avg_buf()[4])
        self.assertEqual(np.average(vals[3:6]), q.get_avg_buf()[5])
        
        q.append(vals[6])
        self.assertEqual(np.average(vals[0:3]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[0:4]), q.get_avg_buf()[1])
        self.assertEqual(np.average(vals[0:5]), q.get_avg_buf()[2])
        self.assertEqual(np.average(vals[1:6]), q.get_avg_buf()[3])
        self.assertEqual(np.average(vals[2:7]), q.get_avg_buf()[4])
        self.assertEqual(np.average(vals[3:7]), q.get_avg_buf()[5])
        self.assertEqual(np.average(vals[4:7]), q.get_avg_buf()[6])
        
        q.append(vals[7])
        self.assertEqual(np.average(vals[0:3]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[0:4]), q.get_avg_buf()[1])
        self.assertEqual(np.average(vals[0:5]), q.get_avg_buf()[2])
        self.assertEqual(np.average(vals[1:6]), q.get_avg_buf()[3])
        self.assertEqual(np.average(vals[2:7]), q.get_avg_buf()[4])
        self.assertEqual(np.average(vals[3:8]), q.get_avg_buf()[5])
        self.assertEqual(np.average(vals[4:8]), q.get_avg_buf()[6])
        self.assertEqual(np.average(vals[5:8]), q.get_avg_buf()[7])
        
        q.append(vals[8])
        self.assertEqual(np.average(vals[0:3]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[0:4]), q.get_avg_buf()[1])
        self.assertEqual(np.average(vals[0:5]), q.get_avg_buf()[2])
        self.assertEqual(np.average(vals[1:6]), q.get_avg_buf()[3])
        self.assertEqual(np.average(vals[2:7]), q.get_avg_buf()[4])
        self.assertEqual(np.average(vals[3:8]), q.get_avg_buf()[5])
        self.assertEqual(np.average(vals[4:9]), q.get_avg_buf()[6])
        self.assertEqual(np.average(vals[5:9]), q.get_avg_buf()[7])
        self.assertEqual(np.average(vals[6:9]), q.get_avg_buf()[8])
        
        q.append(vals[9])
        self.assertEqual(np.average(vals[0:4]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[0:5]), q.get_avg_buf()[1])
        self.assertEqual(np.average(vals[1:6]), q.get_avg_buf()[2])
        self.assertEqual(np.average(vals[2:7]), q.get_avg_buf()[3])
        self.assertEqual(np.average(vals[3:8]), q.get_avg_buf()[4])
        self.assertEqual(np.average(vals[4:9]), q.get_avg_buf()[5])
        self.assertEqual(np.average(vals[5:10]), q.get_avg_buf()[6])
        self.assertEqual(np.average(vals[6:10]), q.get_avg_buf()[7])
        self.assertEqual(np.average(vals[7:10]), q.get_avg_buf()[8])
        
        q.append(vals[10])
        self.assertEqual(np.average(vals[0:5]), q.get_avg_buf()[0])
        self.assertEqual(np.average(vals[1:6]), q.get_avg_buf()[1])
        self.assertEqual(np.average(vals[2:7]), q.get_avg_buf()[2])
        self.assertEqual(np.average(vals[3:8]), q.get_avg_buf()[3])
        self.assertEqual(np.average(vals[4:9]), q.get_avg_buf()[4])
        self.assertEqual(np.average(vals[5:10]), q.get_avg_buf()[5])
        self.assertEqual(np.average(vals[6:11]), q.get_avg_buf()[6])
        self.assertEqual(np.average(vals[7:11]), q.get_avg_buf()[7])
        self.assertEqual(np.average(vals[8:11]), q.get_avg_buf()[8])

if __name__ == '__main__':
    unittest.main()
