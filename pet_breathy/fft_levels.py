
import enum

class FftLevel(enum.IntEnum):
    LEVEL_1 = 0 #enum.auto()   # <- Smallest FFT
    LEVEL_2 = 1 #enum.auto()
    LEVEL_3 = 2 #enum.auto()
    LEVEL_4 = 3 #enum.auto()
    LEVEL_5 = 4 #enum.auto()
    LEVEL_6 = 5 #enum.auto()   # <- Largest FFT

'''
Example:

fps         30
decimation  6

LEVEL_1     bpm range: [18.75 131.25], accum: 3.2 secs
LEVEL_2     bpm range: [9.375 140.625], accum: 6.4 secs
LEVEL_3     bpm range: [4.6875 145.3125], accum: 12.8 secs
LEVEL_4     bpm range: [2.34375 147.65625], accum: 25.6 secs
LEVEL_5     bpm range: [1.171875 148.828125], accum: 51.2 secs
LEVEL_6     bpm range: [0.5859375 149.4140625], accum: 102.4 secs

Where bpm range = [min, max]. min is the minimum breath per minute this fft
size can represent. max is the maximum breath per minute this fft size can
represent. accum is the amount of time it takes to accumulate enough samples to
take the fft.
'''
