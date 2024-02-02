
def is_pow_of_2(v: int) -> bool:
    return (v & (v - 1)) == 0

def get_largest_pow_of_2(v: int) -> int:
    ''' Got this from https://www.geeksforgeeks.org/highest-power-2-less-equal-given-number/
    I want the largest power of 2 that is equal to or less than count
    
    For example:
    - input: 3, output: 2
    - input: 4, output: 4
    - input: 5, output: 4
    - input: 9, output: 8
    '''
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v ^ (v >> 1)
