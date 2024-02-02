import numpy as np

from typing import Any

class FixedBuffer:
    def __init__(self, shape, dtype: Any):
        self.capacity = _get_buf_len_from_shape(shape)
        self.size = 0
        
        self.buf = np.empty(shape, dtype=dtype)

    def __str__(self) -> str:
        return '%s' % self.get_buf()

    def get_size(self) -> int:
        return self.size

    def get_capacity(self) -> int:
        return self.capacity

    def get_buf(self):
        if self.size < self.capacity:
            return self.buf[:self.size]
        return self.buf

    def get_latest_buf(self, count: int) -> np.ndarray:
        if count > self.size:
            raise Exception('Cannot get latest buffer - count: %s, size: %s' % (
                count, self.size))
        return self.buf[self.size - count:self.size]

    def clear_no_erase(self):
        self.size = 0

    def copy(self, new_points: np.ndarray):
        if len(new_points) != self.size:
            raise Exception('Cannot copy points')
        
        np.copyto(self.get_buf(), new_points)

class FixedArray(FixedBuffer):
    def append(self, val: Any):
        if self.size < self.capacity:
            self.buf[self.size] = val
            self.size += 1
        else:
            raise Exception('Cannot append, array is full')

class FixedQueue(FixedBuffer):
    ''' Sort've a queue ... not really, if full, the oldest element, element 0,
    is removed, then the new value is appended to the end
    '''
    def append(self, val: Any):
        self._append(val)

    def _append(self, val: Any):
        if self.size < self.capacity:
            self.buf[self.size] = val
            self.size += 1
        else:
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = val


class FixedAvgQueue(FixedQueue):
    def __init__(self, shape, dtype: Any, kernel_size: int):
        super().__init__(shape, dtype)
        
        if (kernel_size % 2) == 0 or kernel_size <= 1:
            raise Exception('Kernel size needs to be odd and greater than 1')
        
        kernel_data_shape = _update_shape_buf_len(shape, kernel_size)
        
        self.kernel_data = FixedQueue(kernel_data_shape, dtype)
        self.average_data = FixedQueue(shape, dtype)
        
        self.kernel_size = kernel_size
        self.kernel_half_size = kernel_size // 2 + 1

    def clear_no_erase(self):
        self.size = 0
        self.kernel_data.clear_no_erase()
        self.average_data.clear_no_erase()

    def get_avg_buf(self) -> np.ndarray:
        return self.average_data.get_buf()

    def get_latest_avg_buf(self, count: int) -> np.ndarray:
        return self.average_data.get_latest_buf(count)

    def get_act_buf(self) -> np.ndarray:
        return self.get_buf()

    def get_latest_act_buf(self, count: int) -> np.ndarray:
        return self.get_latest_buf(count)

    def append(self, val: Any):
        self._append(val)
        self.average_data.append(val)
        self.kernel_data.append(val)
        
        start_idx = max(0, self.size - self.kernel_half_size)
        
        kernel_data_buf = self.kernel_data.get_buf()
        
        kernel_idx = self.kernel_data.get_size() - self.kernel_size
        
        average_buf = self.average_data.get_buf()
        for idx in range(start_idx, self.size):
            if kernel_idx <= 0:
                average_buf[idx] = np.average(kernel_data_buf)
            else:
                average_buf[idx] = np.average(kernel_data_buf[kernel_idx:])
            kernel_idx += 1

def _get_buf_len_from_shape(shape: tuple):
    '''Assume tuple has two forms:
    - 123
    - (123, ...)
    If an integer, that's it, that's the length. If it's a tuple of 1 or more
    in length, then the length is the first element in the tuple.
    '''
    if type(shape) is int:
        buf_len = shape
    else:
        buf_len = shape[0]
    return buf_len

def _update_shape_buf_len(shape: tuple, new_buf_len: int):
    if type(shape) is int:
        new_shape = new_buf_len
    else:
        new_shape = (new_buf_len,) + shape[1:]
    return new_shape
