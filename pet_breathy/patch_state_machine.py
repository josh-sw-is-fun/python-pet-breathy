from pet_breathy.signal import Signal
from pet_breathy.signal_score import SignalScore
from pet_breathy.fft_levels import FftLevel
from pet_breathy.signal_analyzer_result import SignalAnalyzerResult
from pet_breathy import signal_analyzer

import enum
import numpy as np
import scipy as sp

class SignalLockStateMachine:
    def __init__(self, min_lost_lock_count):
        self.min_lost_lock_count = min_lost_lock_count
        self.score = SignalScore()
        self.reset()

    def reset(self):
        self.signal_lock = False
        self.lost_signal_lock_count = 0
        self.score.reset()

    def has_signal_lock(self):
        return self.signal_lock

    def got_signal_lock(self, fft_level, sig_strength):
        if self.signal_lock:
            self.score.got_lock(fft_level, sig_strength)
        self.signal_lock = True
        self.lost_signal_lock_count = 0

    def lost_signal_lock(self):
        if self.signal_lock:
            self.lost_signal_lock_count += 1
            if self.lost_signal_lock_count >= self.min_lost_lock_count:
                self.reset()
            else:
                self.score.lost_signal_lock()

    def is_signal_lost(self):
        return self.score.is_signal_lost()

class PatchStateMachine:
    class State(enum.Enum):
        COLLECTING = enum.auto()
        COLLECTING_FOR_TIME_DOMAIN_ANALYSIS = enum.auto()
        ANALYSIS = enum.auto()
        FAILED = enum.auto()

    def __init__(self, patch_id, patch_segs, sig_info):
        '''
        @param segs List of Segment objects
        '''
        self.patch_id = patch_id
        self.patch_segs = patch_segs
        self.sig_info = sig_info
        
        self.debug_patch = False
        self.debug_prints = False
        # Blaze
        # - Good points: 38, 12, 16 (once it hits 128 points it turns bad)
        # - Bad points:  53, 32
        #self.debug_patch = self.patch_id == 16 or self.patch_id == 53
        # Mr. Pups
        # - Good points: 83, 51
        # - Bad points:  5, 52
        #self.debug_patch = self.patch_id == 52 or self.patch_id == 51
        # 20231005_094642_sophie.mp4
        # - Good points: 26
        # - Bad points:  25
        #self.debug_patch = self.patch_id == 25 or self.patch_id == 26
        
        # Using LEVEL_3 for the max number of samples we'll collect before failing. Decimation of 6, FPS of 30, LEVEL_3 is 64 samples or FFT size of 64. If the true bpm is around 18 breaths per minute and I need 2 of them to do a time domain analysis, that equal 60 / 18 * 2 = 6.667 seconds to collect. LEVEL_2 or FFT of 32 gives me only 6.4 second window, which "might" work. To be safe I want the next largest FFT. 64 FFT gives me 12.8 second window wich sould be enough to collect at least 2 breaths a of a slow/small bpm.
        self.min_samples_to_analyze_before_fail = \
            signal_analyzer.convert_fft_level_to_num_samples(
                self.sig_info.decimation, FftLevel.LEVEL_3)
        
        # When we have lock on a signal in both time & frequency domain
        self.sig_lock_sm = SignalLockStateMachine(self.sig_info.get_decimated_fps())
        
        self.debug_best_signal = None
        
        self.reset()

    def process(self, signals, yf_points):
        '''
        seg_signals[0] is a list of Signal objects for the center segment
        @param seg_signals List of 9 lists with 2 elements, [0] is a list
            of Signal objects, [1] is yf_avg.
        @param sig_info
        '''
        self.got_good_freq_time_domain = False
        if self.state == PatchStateMachine.State.FAILED:
            return

        self.signals = signals
        self.yf_avg = yf_points
        self._run_current_state()
    
    def has_signal_lock(self):
        return self.sig_lock_sm.has_signal_lock()

    def reset(self):
        self.got_lock_once = False
        self.got_good_freq_time_domain = False
        self.state = PatchStateMachine.State.COLLECTING
        self.sig_lock_sm.reset()
        self.debug_best_signal = None

    def get_state(self):
        return self.state

    def get_score(self):
        return self.sig_lock_sm.score

    def has_had_signal_lock_once(self):
        return self.got_lock_once

    def has_good_freq_time_domain(self):
        return self.got_good_freq_time_domain

    def failed(self):
        return self.state == PatchStateMachine.State.FAILED
    
    def get_debug_signal(self):
        return self.debug_best_signal

    def set_debug_prints(self, enabled):
        pass

    def _run_current_state(self):
        match self.state:
            case PatchStateMachine.State.COLLECTING:
                self._collecting()
            case PatchStateMachine.State.COLLECTING_FOR_TIME_DOMAIN_ANALYSIS:
                self._collecting_for_time_domain_analysis()
            case PatchStateMachine.State.ANALYSIS:
                self._analysis()
            case PatchStateMachine.State.FAILED:
                self._failed()

    def _run_state(self, state):
        self.state = state
        self._run_current_state()
    
    def _collecting(self):
        if self._signal_lost():
            self._lost_signal()
            return
        self._run_state(PatchStateMachine.State.COLLECTING_FOR_TIME_DOMAIN_ANALYSIS)
    
    def _collecting_for_time_domain_analysis(self):
        if self._signal_lost():
            self._lost_signal()
            return
        fft_level = self._get_max_fft_level()
        
        if fft_level <= FftLevel.LEVEL_2:
            if self._can_do_time_domain_analysis():
                self._run_state(PatchStateMachine.State.ANALYSIS)
            # else - stay in this state
        else:
            self._run_state(PatchStateMachine.State.ANALYSIS)
    
    def _analysis(self):
        if self._signal_lost():
            self._lost_signal()
            return
        # This will be the signals at the center patch
        # ... and maybe calling a separate function with the patch things in case we want to run this state machine on other patches
        center_signals = self.signals
        
        good_signals = [ ]
        bpms = [ ]
        signal_guesses = [ ]
        
        if self.debug_patch:
            print('=== patch id:', self.patch_id, '- num sigs', len(center_signals))
            for sig in center_signals:
                print('    - ', sig)

        best_sig_strength = -999
        self.debug_best_signal = None
        
        for signal in center_signals:
            fft_index, min_bpm = self._get_fft_index(signal)
            
            # Need to add 1 to the index which will set the peak_dist a little
            # lower so we have a chance of finding the peaks at fft_index. For
            # example if peak_dist is 16, I have observed some data where the
            # actual peak distance is 15.
            fft_index_threshold = fft_index + 1
            
            peak_dist = int(60 / (fft_index_threshold * min_bpm) * (self.sig_info.fps / self.sig_info.decimation))
            
            peaks = [ ]
            if peak_dist > 0:
                peaks, _ = sp.signal.find_peaks(
                    self.yf_avg,
                    distance=peak_dist,
                    height=signal_analyzer.SIGNAL_STRENGTH_THRESHOLD)
            
            if len(peaks) >= 2:
                avg_peaks = [ ]
                for i in range(1, len(peaks)):
                    avg_peaks.append(peaks[i] - peaks[i - 1])
                avg_peak_dist = np.average(avg_peaks)
                bpm = self._convert_dist_to_bpm(avg_peak_dist)
                std = np.std(avg_peaks)
                
                good_bpm = False
                good_bpm_strength = 0
                
                if self._bpm_in_range(bpm, min_bpm, fft_index):
                    bpms.append(bpm)
                    good_signals.append(signal)
                    good_bpm = True
                    good_bpm_strength = signal.strength
                    if signal.strength > best_sig_strength:
                        best_sig_strength = signal.strength
                        self.debug_best_signal = signal
                        self.debug_best_signal.bpm_est = bpm
                
                if self.debug_patch:
                    print('  - Num peaks:', len(peaks), ' min bpm:', min_bpm, 'fft_index:', fft_index, 'bpm (dist):', bpm, ' o', good_bpm, '%.3f' % good_bpm_strength, 'std:', std)
        
        if bpms:
            self._good_frequency_and_time_domain_signal()
        else:
            self._lost_signal()

    def _good_frequency_and_time_domain_signal(self):
        self.got_lock_once = True
        self.got_good_freq_time_domain = True
        self.sig_lock_sm.got_signal_lock(self.debug_best_signal.fft_level, self.debug_best_signal.strength)

    def _lost_signal(self):
        if len(self.yf_avg) >= self.min_samples_to_analyze_before_fail:
            self.sig_lock_sm.lost_signal_lock()
            
            if not self.got_lock_once or self.sig_lock_sm.is_signal_lost():
                self._run_state(PatchStateMachine.State.FAILED)

    def _failed(self):
        if self.debug_prints:
            print('=== patch id:', self.patch_id, 'FAILED')

    def _signal_lost(self) -> bool:
        return not self.signals
    
    def _get_max_fft_level(self) -> FftLevel:
        max_level = -1
        for signal in self.signals:
            if signal.fft_level > max_level:
                max_level = signal.fft_level
        if max_level == -1:
            raise RuntimeError('Expecting 1 or more signals to get max fft level')
        return max_level
    
    def _can_do_time_domain_analysis(self):
        '''
        signal info
        - Need to check if the signal contains enough data points to contain 2 periods given the lowest signal
        
        self.signals
        
        signal
        - fft_size
        - bpm_est
        
        bpm_est = 70 bpm
        
        Need
        - decimation
        - fps
        
        16 points to 31 points, could use all 31 points when calculating
        
        30/6 = 5.0
        5 frames per second
        16/5 = 3.2 seconds worth of points (fft_seconds)
        
60/70 = 0.8571428571428571 breaths per second
        
        0.8571428571428571 * 2 <= 3.2
        1.7142857142857142 <= 3.2       True
        
bpm_est = 18
60/18 = 3.3333333333333335 breaths per second

        3.3333333333333335 * 2 <= 3.2
        6.666666666666667 <= 3.2        False

32/5 = 6.4 seconds
64/5 = 12.8 seconds

To make a time domain estimate on 18 bpm frequency domain estimate, we need 6.6667 seconds worth of data
The 32 FFT almost makes it, 64 FFT is more than enough to make the time domain estimate, meaning it
will contain 2 or more peaks

        If fft_size is 16, the bpm_est is
        '''
        center_signals = self.signals
        
        if not center_signals:
            raise RuntimeError('Cannot check if time domain analysis can be '
                'performed, expecting signals to not be empty')
        
        min_bpm = 99999
        for signal in center_signals:
            if signal.bpm_est < min_bpm:
                min_bpm = signal.bpm_est
        
        #print('id: %s, min bpm: %s' % (self.patch_id, min_bpm))
        #for signal in center_signals:
        #    print('- %s' % signal)
        # Can use these to pass to signal_analyzer to do the calculation
        actual_fps = self.sig_info.fps / self.sig_info.decimation
        fft_seconds = center_signals[0].fft_size / actual_fps
        # breaths per second
        bps = 60.0 / min_bpm
        # Need at least 2 breaths in the data to do a time domain analysis
        time_needed = bps * 2
        
        return time_needed <= fft_seconds
    
    def _get_fft_index(self, signal: Signal):
        # TODO Shouldn't need to calculate this so often, need to refactor
        min_bpm = self._calc_min_bpm(signal.fft_size)
        return int(signal.bpm_est / min_bpm), min_bpm
    
    def _calc_min_bpm(self, fft):
        return (1 / (fft / (self.sig_info.fps / self.sig_info.decimation))) * 60
    
    def _convert_dist_to_bpm(self, avg_peak_dist):
        return 60 / (avg_peak_dist * (self.sig_info.decimation / self.sig_info.fps))
    
    def _bpm_in_range(self, bpm, min_bpm, fft_index):
        ''' Is bpm in the range described above like [fft-1 fft+1] '''
        min_fft_index = (fft_index - 1) if fft_index > 1 else .5
        max_fft_index = fft_index + 1
        result = (min_fft_index * min_bpm) <= bpm <= (max_fft_index * min_bpm)
        #print('     %5s %s <= %s <= %s, fft_index: %s, min_bpm: %s' % (
        #    result,
        #    min_fft_index * min_bpm,
        #    bpm,
        #    max_fft_index * min_bpm,
        #    fft_index,
        #    min_bpm))
        return result
