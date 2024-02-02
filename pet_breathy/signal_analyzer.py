from pet_breathy.signal import Signal
from pet_breathy.signal_info import SignalInfo
from pet_breathy.signal_analyzer_result import SignalAnalyzerResult
from pet_breathy.fft_levels import FftLevel
from pet_breathy.fft_overlapper import FftOverlapper
from pet_breathy import utils

import scipy as sp
import numpy as np

# Arbitrarily chosen, use this to filter points that have a weak
# signal. Points/segments that have a weak signal can be moved
# elsewhere that might have a better signal
#
# Notes:
# .0005 seemed to work for fft 128 (fft level 3) but since I limited it to fft
# level 0, 1, 2, I put the threshold to 0.001. Could be that averaging isn't
# needed when fft is 128
SIGNAL_STRENGTH_THRESHOLD = 0.001

LINSPACE_LOOKUP = { }
for fft in range(4, 13):
    fft_len = 2**fft
    LINSPACE_LOOKUP[fft_len] = np.linspace(0, fft_len, fft_len)

class SignalAnalyzer:
    def __init__(self, fps: int, decimation: int):
        self.debug = False

        if fps != 30:
            raise Exception('SignalAnalyzer - Expects 30 FPS')
        
        if (fps % decimation) != 0:
            raise Exception('SignalAnalyzer - FPS should be evenly divisible by decimation')
        
        try:
            info = _g_fft_info.lookup[decimation]
        except KeyError:
            raise Exception('SignalAnalyzer - Decimation needs to be one of the following: 1, 2, 3, 5, 6')
        fft_sizes = list(info['fft_sizes'])
        avg_kernel_size = info['avg_kernel_size']
        
        self.fft_lookup = dict(zip(fft_sizes, [e for e in FftLevel]))
        if self.debug:
            print('fft lookup: %s' % self.fft_lookup)
        
        decimated_fps = fps // decimation
        decimated_fps_time = 1.0 / decimated_fps
        
        # Use xffts to plot ffts
        xffts = [ ]
        for fft_size in fft_sizes:
            xfft = sp.fft.fftfreq(fft_size, decimated_fps_time)[:fft_size//2]
            xffts.append(xfft)
        
        self.fps = fps
        self.decimation = decimation
        self.avg_kernel_size = avg_kernel_size
        self.decimated_fps = decimated_fps
        self.decimated_fps_time = decimated_fps_time
        
        self.fft_sizes = fft_sizes
        self.xffts = xffts
        
        self.info = SignalInfo()
        self.info.fps = self.fps
        self.info.decimation = self.decimation
        
        if self.debug:
            print('Signal analyzer info:')
            print('- FPS:           %s' % self.fps)
            print('- Decimation:    %s' % self.decimation)
            print('- Decimated fps: %s' % self.decimated_fps)
            print('- FFT info:')
        
        # Key:      FFT size
        # Value:    BPM precision
        self.bpm_fft_precisions = { }
        for fft in self.fft_sizes:
            min_idx = 1
            max_idx = fft // 2 - 1
            min_bpm = (min_idx / (fft / (self.fps / self.decimation))) * 60
            max_bpm = (max_idx / (fft / (self.fps / self.decimation))) * 60
            # Amount of time it would take to accumulate fft worth of points
            accum_secs = fft / (self.fps / self.decimation)
            
            if self.debug:
                print('  - FFT size: %s, BPM range: [%.2f %.2f], time: %.2f' % (
                    fft, min_bpm, max_bpm, accum_secs))
            
            self.bpm_fft_precisions[fft] = min_bpm

            if self.debug:
                self.debug_fft_fh = open('./debug/fft.json', 'w')
        
        if self.debug:
            print('BPM precisions: %s' % self.bpm_fft_precisions)
    
    def get_min_fft_size(self) -> int:
        return self.fft_sizes[0]
    
    def get_max_fft_size(self) -> int:
        return self.fft_sizes[-1]
    
    def get_avg_kernel_size(self) -> int:
        return self.avg_kernel_size
    
    def get_signal_info(self) -> SignalInfo:
        return self.info
    
    def calc_avg_signals(self,
            ys: list[np.ndarray],
            overlapper: FftOverlapper) -> SignalAnalyzerResult:
        if self.debug:
            for y in ys:
                self.debug_fft_fh.write('%s\n' % y)
            self.debug_fft_fh.write('\n')
        
        yf_avg = np.asarray(ys[0], dtype=float)
        for i in range(1, len(ys)):
            yf_avg += ys[i]
        yf_avg /= float(len(ys))
        
        overlapper.add_vals(yf_avg)
        
        yf_fft = overlapper.get_spectra()
        
        # width=(1,3)
        # Using width seemed to work in stand alone testing but didn't seem to be good enough weeding false positives
        peaks = self.find_peaks(
            yf_fft,
            height=SIGNAL_STRENGTH_THRESHOLD)
        
        return SignalAnalyzerResult(
            self.calc_signals(ys, yf_fft, peaks), yf_avg)

    def calc_signals(self, ys: list[np.ndarray], yf_fft: np.ndarray, peaks: np.ndarray) -> list[Signal]:
        signals = [ ]
        fft_size = len(ys[0])
        
        for fft_idx in peaks:
            signal = Signal()
            signal.strength = yf_fft[fft_idx]
            signal.bpm_est = (fft_idx / (fft_size / (self.fps / self.decimation))) * 60
            signal.fft_size = fft_size
            signal.bpm_precision = self.bpm_fft_precisions[fft_size]
            signal.fft_level = self.fft_lookup[signal.fft_size]
            signal.decimation = self.decimation
            
            '''
            # Noticed that some signals straddle fft bins. If there's an
            # adjacent bin, I want that to be considered when calculated bpm
            # since it will effect the min/max range.
            prev_fft_idx = max(fft_idx - 1, 1)
            next_fft_idx = min(fft_idx + 1, len(yf_fft) - 1)
            if prev_fft_idx != fft_idx:
                if yf_fft[prev_fft_idx] * .5 >= SIGNAL_STRENGTH_THRESHOLD and \
                        (yf_fft[prev_fft_idx] / yf_fft[fft_idx]) >= 0.75:
                    signal.prev_fft_level_has_energy = True
            if next_fft_idx != fft_idx:
                if yf_fft[next_fft_idx] * .5 >= SIGNAL_STRENGTH_THRESHOLD and \
                        (yf_fft[next_fft_idx] / yf_fft[fft_idx]) >= 0.75:
                    signal.next_fft_level_has_energy = True
            '''
            signals.append(signal)
        
        return signals

    def find_peaks(self, yf: np.ndarray, height: float) -> np.ndarray:
        peaks, _ = sp.signal.find_peaks(yf, height=height)
        return peaks

    def calc_avg_fft(self, ys: list[np.ndarray]) -> np.ndarray:
        yf_avg = self._calc_fft(ys[0])
        for i in range(1, len(ys)):
            yf_avg += self._calc_fft(ys[i])
        return yf_avg

    def get_xfft(self, fft_size) -> np.ndarray:
        fft_idx = self.fft_sizes.index(fft_size)
        return self.xffts[fft_idx]

    '''
        # (11 / (32 / (30 / 3))) * 60   = bpm
        #  ^^    ^^    ^^   ^      ^^
        #   |     |     |   |       `- 60 seconds
        #   |     |     |   `--------- Decimation factor
        #   |     |     `------------- Original FPS
        #   |     `------------------- N, FFT size
        #   `------------------------- index using fft bin increment size and the point x-coord
    '''
    
    def _calc_fft(self, y: np.ndarray) -> np.ndarray:
        yf = sp.fft.fft(y)
        yf = 2.0 / len(y) * np.abs(yf[:len(y) // 2])
        return yf

def fit_sin_with_guess(tt, yy, guess_freq):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    try:
        popt, pcov = sp.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=100)
    except RuntimeError:
        return None
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return fitfunc(tt)

def convert_fft_level_to_num_samples(decimation: int, level: FftLevel) -> int:
    return _g_fft_info.lookup[decimation]['fft_sizes'][int(level)]

def convert_num_samples_to_fft_level(decimation: int, num_samples) -> FftLevel:
    largest_pow_2 = utils.get_largest_pow_of_2(num_samples)
    fft_sizes = _g_fft_info.lookup[decimation]['fft_sizes']
    for idx, fft_size in enumerate(fft_sizes):
        if fft_size == largest_pow_2:
            return FftLevel(idx)
    return None

class _FftInfo:
    def __init__(self):
        '''
        Lookup table
        Key:    decimation size
        Value:  In related to the decimation

        The number of fft_sizes elements must equal FftLevel number of levels

        Used calc_bpm_range_and_runtime to get the following numbers
        I wanted the minimum runtime to collect samples to be 3 or greater seconds
        '''
        self.lookup = {
            1: {
                'fft_sizes': [ 128, 256, 512, 1024, 2048, 4096 ],
                'avg_kernel_size': 31,
            },
            2: {
                'fft_sizes': [ 64, 128, 256, 512, 1024, 2048 ],
                'avg_kernel_size': 15,
            },
            3: {
                'fft_sizes': [ 32, 64, 128, 256, 512, 1024 ],
                'avg_kernel_size': 11,
            },
            5: {
                'fft_sizes': [ 32, 64, 128, 256, 512, 1024 ],
                'avg_kernel_size': 9,
            },
            6: {
                # Commented out 128, 256, 512 ffts
                # Bunch more testing is needed to get these ffts to work. 64
                # seems to work pretty well for all test data
                # Larger fft would provide higher precision in bpm estimation ...
                'fft_sizes': [ 16, 32, 64 ], #, 128, 256, 512 ],

                # Smaller kernel size of 7 is more responsive than 13
                # But some videos are better characterized with a value of 13 vs 7
                # Too small or too large the signal isn't detected as well. The
                # averaging is meant to help remove the overall movement of the
                # points over time but not too responsive that we lose too much
                # signal strength. Unresponsive means that the points could drift
                # down or up, this will cause the FFT to detect a large spike of
                # enery at the 1st FFT bin.
                'avg_kernel_size': 9, #7, #13 #15
            },
        }

_g_fft_info = _FftInfo()

'''
min_fft_size = 16
max_fft_size = 256
for i in range(100):
    fft_size = 1 << i
    if min_fft_size <= fft_size <= max_fft_size:
        print(fft_size)
    elif fft_size > max_fft_size:
        break
'''

'''
def calc_bpm_range_and_runtime(fft, decimation):
    fps = 30
    min_idx = 1
    max_idx = fft // 2 - 1
    min_bpm = (min_idx / (fft / (fps / decimation))) * 60
    max_bpm = (max_idx / (fft / (fps / decimation))) * 60
    # accum_secs - Amount of time it would take to collect enough samples
    accum_secs = fft / (fps / decimation)
    print('bpm range: [%s %s], accum: %s secs' % (min_bpm, max_bpm, accum_secs))

calc_bpm_range_and_runtime(256, 6)
'''
