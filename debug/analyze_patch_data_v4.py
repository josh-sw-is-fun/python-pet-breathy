from pet_breathy.debug_things import DebugPatchAnalyzer, DebugPatchAnalyzerV2
from pet_breathy.fixed_data_structures import FixedAvgQueue
from pet_breathy.video_file_reader import VideoFileReader
from pet_breathy.video_display import VideoDisplay
from pet_breathy.segment import Segment
from pet_breathy.patch_pow_2_info import PatchPow2Info
from pet_breathy import patch_pow_2_info
from pet_breathy import utils
from pet_breathy.signal import Signal
from pet_breathy.fft_levels import FftLevel
import sys
import os
import json
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

import scipy as sp
import cv2 as cv

'''
PYTHONPATH=$(pwd) python3

exec(open('./debug/analyze_patch_data_v4.py').read())
load_analyzer_v2()

'''

def load_analyzer_v2():
    #fn = './debug/data/PXL_20230825_040038487.json'
    fn = './debug/data/Blaze2.json'
    
    global analyzer
    analyzer = DebugPatchAnalyzerV2(fn)
    
    print('File name: %s' % fn)
    print('Fps:       %s' % analyzer.video_info.fps)
    print('Dim:       %s x %s (width x height)' % (analyzer.video_info.width, analyzer.video_info.height))
    print('Frames:    %s' % analyzer.video_info.frame_count)


def run():
    global analyzer
    sim = BreathySim(analyzer)
    sim.run()

class PatchStat:
    def __init__(self, patch_id):
        self.patch_id = patch_id
        self.largest_fft_out_of_range = False
        self.frames = [ ]
        self.fft_size = 0
        self.seg_offset = 0
        self.analysis_enabled = True
        self.analysis_disabled_where = ''

    def demote_fft(self):
        self.fft_size //= 2
        self.seg_offset += self.fft_size
        if self.fft_size < 16:
            self.analysis_enabled = False

    def promote_fft(self):
        self.fft_size *= 2

    def can_analyze(self):
        return self.analysis_enabled

    def disable_analysis(self, where):
        self.analysis_enabled = False
        self.analysis_disabled_where = where

    def add_patch_frame(self, patch_frame_stat):
        if patch_frame_stat.largest_fft_out_of_range:
            self.largest_fft_out_of_range = True
        self.frames.append(patch_frame_stat)

class PatchFrameStat:
    def __init__(self, frame_num):
        self.frame_num = frame_num
        self.movable_largest_pow_2 = -1
        self.largest_fft_out_of_range = False
        self.can_do_time_domain_analysis = False
        self.at_least_one_bpm_in_range = False
        self.no_peaks_found = False
        self.bpms_found = 0
        self.sig_infos = [ ]
    
    def add_sig_info(self, dist, bpm, sig):
        self.sig_infos.append(SigInfo(dist, bpm, sig))

class SigInfo:
    def __init__(self, dist, bpm, sig):
        self.dist = dist
        self.bpm = bpm
        self.sig = sig
    def __str__(self):
        return 'dist: %s, bpm: %s, sig: %s' % (self.dist, self.bpm, self.sig)

def look_at_patch_stats():
    global patch_stats
    for patch_id, patch_stat in patch_stats.items():
        patch_id = int(patch_id)
        print('- patch id:', patch_id)
        print('  - largest_fft_out_of_range:    %s' % patch_stat.largest_fft_out_of_range)
        if True or patch_stat.largest_fft_out_of_range or patch_id == 52:
            for frame in patch_stat.frames:
                print('    - frame %s, '
                    'out of range: %s, '
                    'pow 2: %s, '
                    'can do: %s, '
                    'bpms: %s' % (
                    frame.frame_num,
                    frame.largest_fft_out_of_range,
                    frame.movable_largest_pow_2,
                    frame.can_do_time_domain_analysis,
                    frame.bpms_found))
                for sig_info in frame.sig_infos:
                    print('     - sig info - %s' % sig_info)

class BreathySim:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        print('Num of steps: %s' % len(self.analyzer.patch_steps))
        
        # Key:   patch_id
        # Value: PtachStat
        global patch_stats
        patch_stats = { }
        self.patch_stats = patch_stats

    def run(self):
        for self.step_idx, self.step in enumerate(self.analyzer.patch_steps):
            self.frame_num = int(self.step['frame_num'])
            print('Frame #', self.frame_num)
            self._analyze_step()

    def _analyze_step(self):
        #self.analyzer.load_frame(int(self.step['frame_num']))
        
        self.static_patches = self.step['static_patches']
        self.movable_patches = self.step['movable_patches']
        
        for self.patch_id, self.movable_patch in self.movable_patches.items():
            try:
                self.patch_stat = self.patch_stats[self.patch_id]
            except KeyError:
                self.patch_stat = PatchStat(self.patch_id)
                self.patch_stats[self.patch_id] = self.patch_stat
            
            if self.patch_stat.can_analyze():
                self.patch_frame_stat = PatchFrameStat(self.frame_num)
                self._analyze_patch()
                self.patch_stat.add_patch_frame(self.patch_frame_stat)
    
    def _analyze_patch(self):
        #self.center_point = self.movable_patch['points'][0]
        #self.analyzer.draw_point(self.patch_id, self.center_point[0], self.center_point[1])
        #self.analyzer.render_frame()
        
        self.seg = self.movable_patch['segs'][0]
        self._analyze_segment()
    
    def _analyze_segment(self):
        try:
            seg_avg = self.seg['avg_y_coords'][self.patch_stat.seg_offset:]
            seg_act = self.seg['act_y_coords'][self.patch_stat.seg_offset:]
        except TypeError:
            print('self.patch_stat.seg_offset: %s, len: %s' % (self.patch_stat.seg_offset, len(self.seg['avg_y_coords'])))
            raise
        
        #print('Patch id: %s, point count: %s' % (self.patch_id, len(seg_avg)))
        movable_largest_pow_2 = \
            utils.get_largest_pow_of_2(len(seg_avg))
        #print('Patch id: %s, largest pow 2: %s' % (self.patch_id, movable_largest_pow_2))
        
        self.patch_stat.fft_size = movable_largest_pow_2
        
        self.patch_frame_stat.movable_largest_pow_2 = movable_largest_pow_2
        
        if movable_largest_pow_2 < 16:
            self.patch_frame_stat.largest_fft_out_of_range = True
            # Still accumulating points maybe
            return
        
        sp_pow_2_infos = [ ]
        for self.patch_id, static_patch in self.static_patches.items():
            coords_len = len(static_patch['segs'][0]['avg_y_coords'])
            sp_pow_2 = utils.get_largest_pow_of_2(coords_len)
            
            # self.signal_analyzer.get_min_fft_size()
            # Decimation is 6, min fft size is 16
            if sp_pow_2 < 16:
                continue
            sp_pow_2_infos.append(PatchPow2Info(int(self.patch_id), sp_pow_2))
        
        if not sp_pow_2_infos:
            # Static patches accumulating points maybe
            return
        
        best_sp_idxs = patch_pow_2_info.find_best_infos(
            movable_largest_pow_2,
            sp_pow_2_infos)
        
        best_static_patch = self.static_patches[f'{best_sp_idxs[0]}']
        best_static_patch_coords_len = len(best_static_patch['segs'][0]['avg_y_coords'])
        common_pow_2 = min(
            movable_largest_pow_2,
            utils.get_largest_pow_of_2(best_static_patch_coords_len))
        #print('Common pow 2: %s' % self.common_pow_2)
        
        dy_coords = [ ]
        
        for best_sp_idx in best_sp_idxs:
            static_patch = self.static_patches[f'{best_sp_idxs[best_sp_idx]}']
            static_avg_y_coord = _convert_to_np_array(
                static_patch['segs'][0]['avg_y_coords'][-common_pow_2:])
            static_act_y_coord = _convert_to_np_array(
                static_patch['segs'][0]['act_y_coords'][-common_pow_2:])
            
            movable_avg_y_coord = _convert_to_np_array(seg_avg[-common_pow_2:])
            movable_act_y_coord = _convert_to_np_array(seg_act[-common_pow_2:])
            
            avg_diff_y_coord = static_avg_y_coord - movable_avg_y_coord
            dy_coord = (static_act_y_coord - movable_act_y_coord) - avg_diff_y_coord
            
            dy_coords.append(dy_coord)
        
        dy_coord_avg = _convert_to_np_array(dy_coords[0])
        for i in range(1, len(dy_coords)):
            dy_coord_avg += dy_coords[i]
        dy_coord_avg /= float(len(dy_coords))
        
        dy_fft_of_avg = _calc_fft(dy_coord_avg)
        
        SIGNAL_STRENGTH_THRESHOLD = 0.2
        peaks, _ = sp.signal.find_peaks(
            dy_fft_of_avg,
            height=SIGNAL_STRENGTH_THRESHOLD)
        
        signals = [ ]
        fft_size = len(dy_coords[0])
        fps = 30
        decimation = 6
        bpm_fft_precisions = {16: 18.75, 32: 9.375, 64: 4.6875, 128: 2.34375, 256: 1.171875, 512: 0.5859375}
        fft_lookup = {16: FftLevel.LEVEL_1, 32: FftLevel.LEVEL_2, 64: FftLevel.LEVEL_3, 128: FftLevel.LEVEL_4, 256: FftLevel.LEVEL_5, 512: FftLevel.LEVEL_6}
        
        for fft_idx in peaks:
            signal = Signal()
            signal.strength = dy_fft_of_avg[fft_idx]
            signal.bpm_est = (fft_idx / (fft_size / (fps / decimation))) * 60
            signal.fft_size = fft_size
            signal.bpm_precision = bpm_fft_precisions[fft_size]
            signal.fft_level = fft_lookup[signal.fft_size]
            signal.decimation = decimation
            signals.append(signal)
        
        if _can_do_time_domain_analysis(signals):
            #print('Can do time domain analysis')
            self.patch_frame_stat.can_do_time_domain_analysis = True
        else:
            #print('Cannot do time domain analysis')
            self.patch_stat.demote_fft()
            return
        
        peaks_with_peak_dist = np.arange(0)
        bpms = [ ]
        res = None
        for signal in signals:
            #print('--------------------------------------------')
            #print('- Signal: %s' % signal)
            fft_index, min_bpm = _get_fft_index(signal)
            #print('  fft index: %s, min bpm: %s' % (fft_index, min_bpm))
            
            energy_in_prev_fft = False
            energy_in_next_fft = False
            
            prev_fft_index = max(fft_index - 1, 1)
            next_fft_index = min(fft_index + 1, len(dy_fft_of_avg) - 1)
            if prev_fft_index != fft_index:
                frac = dy_fft_of_avg[prev_fft_index] / dy_fft_of_avg[fft_index]
                if dy_fft_of_avg[prev_fft_index] * .5 >= SIGNAL_STRENGTH_THRESHOLD and \
                        frac >= 0.75:
                    #print(' !!!! Additional energy in prev fft bucket, fft: %s, prev fft: %s, frac: %s !!!!' % (dy_fft_of_avg[fft_index], dy_fft_of_avg[prev_fft_index], frac))
                    energy_in_prev_fft = True
            if next_fft_index != fft_index:
                frac = dy_fft_of_avg[next_fft_index] / dy_fft_of_avg[fft_index]
                if dy_fft_of_avg[next_fft_index] * .5 >= SIGNAL_STRENGTH_THRESHOLD and \
                        frac >= 0.75:
                    #print(' !!!! Additional energy in next fft bucket, fft: %s, next fft: %s, frac: %s !!!!' % (dy_fft_of_avg[fft_index], dy_fft_of_avg[next_fft_index], frac))
                    energy_in_next_fft = True
            #min_fft_index = max(fft_index - 1, 1)
            #max_fft_index = min(fft_index + 1 + 1, len(dy_fft_of_avg) - 1)
            #for temp_fft_index in range(min_fft_index, max_fft_index):
            #    print('  - [%d] fft: %s' % (temp_fft_index, dy_fft_of_avg[temp_fft_index]))
            
            fft_index_threshold = fft_index + 1
            
            # BPM   min_bpm * fft_index
            # BPS   BPM / 60.0
            peak_dist = int(60 / (fft_index_threshold  * min_bpm) * (fps / decimation))
            #print('  peak dist: %s' % peak_dist)
            
            #peaks_with_no_peak_dist, _ = sp.signal.find_peaks(dy_coord_avg)
            #print('  peaks with no peak dist: %s, len(peaks): %s' % (peaks_with_no_peak_dist, len(peaks_with_no_peak_dist)))
            if peak_dist > 0:
                peaks_with_peak_dist, _ = sp.signal.find_peaks(
                    dy_coord_avg,
                    distance=peak_dist,
                    height=SIGNAL_STRENGTH_THRESHOLD)
            else:
                continue
            #    print('  --- peak dist is zero')
            #print('  peaks with peak dist: %s, len(peaks): %s' % (peaks_with_peak_dist, len(peaks_with_peak_dist)))
            
            if len(peaks_with_peak_dist) >= 2:
                avg_peaks = [ ]
                for i in range(1, len(peaks_with_peak_dist)):
                    avg_peaks.append(peaks_with_peak_dist[i] - peaks_with_peak_dist[i - 1])
                avg_peak_dist = np.average(avg_peaks)
                bpm = _convert_dist_to_bpm(avg_peak_dist)
                
                tt = np.linspace(0, len(dy_coord_avg), len(dy_coord_avg))
                #tt2 = np.linspace(0, len(dy_coord_avg), len(dy_coord_avg)) # *len(dy_coord_avg))
                
                # guess bpm / ((fps / decimation) * 60) = guess freq
                #guess_freq = bpm / ((fps / decimation) * 60)
                guess_freq = signal.bpm_est / ((fps / decimation) * 60)
                #print('  >>>>>> guess freq', guess_freq)
                res = fit_sin_with_guess(tt, dy_coord_avg, guess_freq)
                #print(res)
                #print('fit curve y-coords', res["fitfunc"](tt2))
                y0 = np.absolute(dy_coord_avg)
                y1 = np.absolute(res["fitfunc"](tt))
                dist = np.sum(y1 - y0)
                
                found_bpm = False
                
                #print('  y0', len(y0), 'y1', len(y1), 'dist', dist)
                
                #print('  avg peak dist: %s, bpm: %s' % (avg_peak_dist, bpm))
                #print('  bpm: %s, min_bpm: %s, fft_index: %s' % (bpm, min_bpm, fft_index))
                #if _bpm_in_range(bpm, min_bpm, fft_index) or signal.strength > 1.25:
                if _bpm_in_range(bpm, min_bpm, fft_index):
                    #print('  ==============> :) BPM is in range')
                    self.patch_frame_stat.at_least_one_bpm_in_range = True
                    bpms.append(bpm)
                    self.patch_frame_stat.add_sig_info(dist, bpm, signal)
                    found_bpm = True
                    #bpms.append(signal.bpm_est)
                    #break
                #else:
                #    print('  BPM is out of range')
                
                if energy_in_prev_fft and not found_bpm:
                    if _bpm_in_range(bpm, min_bpm, fft_index - 1):
                        #print('  ==============> :) BPM of prev bucket is in range')
                        self.patch_frame_stat.at_least_one_bpm_in_range = True
                        bpms.append(bpm)
                        self.patch_frame_stat.add_sig_info(dist, bpm, signal)
                        found_bpm = True
                    #else:
                    #    print('  BPM of prev bucket is out of range')
                
                if energy_in_next_fft and not found_bpm:
                    if _bpm_in_range(bpm, min_bpm, fft_index + 1):
                        #print('  ==============> :) BPM of next bucket is in range')
                        self.patch_frame_stat.at_least_one_bpm_in_range = True
                        bpms.append(bpm)
                        self.patch_frame_stat.add_sig_info(dist, bpm, signal)
                        found_bpm = True
                    #else:
                    #    print('  BPM of next bucket is out of range')
                
                if found_bpm:
                    self.patch_frame_stat.bpms_found = len(bpms)
            else:
                self.patch_frame_stat.no_peaks_found = True
        #print('Num bpms found: %s, %s' % (len(bpms), bpms))
        if not bpms:
            self.patch_stat.demote_fft()


def _convert_to_np_array(vals: list):
    return np.asarray(vals, dtype=float)

def _calc_fft(y: np.ndarray) -> np.ndarray:
    yf = sp.fft.fft(y)
    yf = 2.0 / len(y) * np.abs(yf[:len(y) // 2])
    return yf

def _can_do_time_domain_analysis(signals):
    if not signals:
        #print('No signals to do time domain analysis')
        return False
    
    if not signals:
        raise RuntimeError('Cannot check if time domain analysis can be '
            'performed, expecting signals to not be empty')
    
    min_bpm = 99999
    for signal in signals:
        if signal.bpm_est < min_bpm:
            min_bpm = signal.bpm_est
    
    #print('min bpm: %s' % (min_bpm))
    #for signal in signals:
    #    print('- %s' % signal)
    # Can use these to pass to signal_analyzer to do the calculation
    fps = 30
    decimation = 6
    actual_fps = fps / decimation
    fft_seconds = signals[0].fft_size / actual_fps
    # breaths per second
    bps = 60.0 / min_bpm
    # Need at least breaths in the data to do a time domain analysis
    time_needed = bps * 2
    
    #print('time needed: %s, fft seconds: %s' % (time_needed, fft_seconds))
    
    return time_needed <= fft_seconds

def _get_fft_index(signal: Signal):
    min_bpm = _calc_min_bpm(signal.fft_size)
    return int(signal.bpm_est / min_bpm), min_bpm

def _calc_min_bpm(fft):
    fps = 30.0
    decimation = 6.0
    return (1.0 / (fft / (fps / decimation))) * 60.0

def _convert_dist_to_bpm(avg_peak_dist):
    decimation = 6.0
    fps = 30.0
    return 60.0 / (avg_peak_dist * (decimation / fps))

def fit_sin_with_guess(tt, yy, guess_freq):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    #tt = np.array(tt)
    #yy = np.array(yy)
    #ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    #Fyy = abs(np.fft.fft(yy))
    
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = sp.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def _bpm_in_range(bpm, min_bpm, fft_index):
    ''' Is bpm in the range described above like [fft-1 fft+1] '''
    min_fft_index = (fft_index - 1) if fft_index > 1 else .5
    max_fft_index = fft_index + 1
    #print('is bpm in range? - min: %s, bpm: %s, max: %s' % (
    #    (min_fft_index * min_bpm),
    #    bpm,
    #    (max_fft_index * min_bpm)))
    return (min_fft_index * min_bpm) <= bpm <= (max_fft_index * min_bpm)
