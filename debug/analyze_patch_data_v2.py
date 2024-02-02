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

exec(open('./debug/analyze_patch_data_v2.py').read())
load_analyzer()
load_analyzer_v2()
'''

def load_analyzer():
    global analyzer
    #analyzer = DebugPatchAnalyzer('./debug/data/Blaze2_patch_data_avg7.json')
    #analyzer = DebugPatchAnalyzer('./debug/data/PXL_20230825_040038487_line_patch_data_avg7.json')
    analyzer = DebugPatchAnalyzer('./debug/data/PXL_20230825_040038487.json')

def load_analyzer_v2():
    fn = './debug/data/PXL_20230825_040038487.json'
    #fn = './debug/data/Blaze2.json'
    
    global analyzer
    analyzer = DebugPatchAnalyzerV2(fn)
    
    print('File name: %s' % fn)
    print('Fps:       %s' % analyzer.video_info.fps)
    print('Dim:       %s x %s (width x height)' % (analyzer.video_info.width, analyzer.video_info.height))
    print('Frames:    %s' % analyzer.video_info.frame_count)

def inspect_patch_v2(step_idx, patch_id):
    ''' Interesting patch ids:
    - Good 19
    - Bad  17
    
    Blaze2
    - Good 133
    '''
    global analyzer
    print('Num of steps: %s' % len(analyzer.patch_steps))
    step = analyzer.patch_steps[step_idx]
    
    static_patches = step['static_patches']
    movable_patches = step['movable_patches']
    
    analyzer.load_frame(int(step['frame_num']))
    
    try:
        movable_patch = movable_patches[str(patch_id)]
    except KeyError as err:
        print('Key error: %s' % err)
        print('movable patches len:  %s' % len(movable_patches))
        print('movable patches keys: %s' % movable_patches.keys())
        return
    
    try:
        print('movable_patch keys: %s' % movable_patch.keys())
        center_point = movable_patch['points'][0]
    except KeyError as err:
        #print('movable patch: %s' % movable_patch)
        print('movable patch keys: %s' % movable_patch.keys())
        print('segs len: %s' % len(movable_patch['segs']))
        return
    
    try:
        print('Point to render: %s, %s' % (center_point[0], center_point[1]))
        analyzer.draw_point(patch_id, center_point[0], center_point[1])
        analyzer.render_frame()
    except KeyError as err:
        print('Error ? %s' % err)
        print('center_point: %s' % center_point)
        return
    
    center_seg = movable_patch['segs'][0]
    center_seg_avg = center_seg['avg_y_coords']
    center_seg_act = center_seg['act_y_coords']
    # Keys: avg_y_coords, act_y_coords
    print('point count: %s' % len(center_seg_avg))
    
    movable_largest_pow_2 = \
            utils.get_largest_pow_of_2(len(center_seg_avg))
    print('Largest pow 2: %s' % movable_largest_pow_2)
    
    sp_pow_2_infos = [ ]
    for patch_id, static_patch in static_patches.items():
        coords_len = len(static_patch['segs'][0]['avg_y_coords'])
        sp_pow_2 = utils.get_largest_pow_of_2(coords_len)
        
        # self.signal_analyzer.get_min_fft_size()
        # Decimation is 6, min fft size is 16
        if sp_pow_2 < 16:
            continue
        sp_pow_2_infos.append(PatchPow2Info(int(patch_id), sp_pow_2))
    
    best_sp_idxs = patch_pow_2_info.find_best_infos(
        movable_largest_pow_2,
        sp_pow_2_infos)
    print('>>> best sp idx: %s, movable largest pow of 2: %s' % (best_sp_idxs, movable_largest_pow_2))
    for i in best_sp_idxs:
        print('- [%s] %s' % (i, sp_pow_2_infos[i]))
    
    best_static_patch = static_patches[f'{best_sp_idxs[0]}']
    best_static_patch_coords_len = len(best_static_patch['segs'][0]['avg_y_coords'])
    common_pow_2 = min(
        movable_largest_pow_2,
        utils.get_largest_pow_of_2(best_static_patch_coords_len))
    print('Common pow 2: %s' % common_pow_2)
    
    # def _analyze_movable_seg(self,
    #    movable_seg: Segment,
    #    static_patches: list[StaticPatch],
    #    best_sp_idxs: list[int],
    #    common_pow_2: int) -> list[Signal]:
    dy_coords = [ ]
        
    for best_sp_idx in best_sp_idxs:
        static_patch = static_patches[f'{best_sp_idxs[best_sp_idx]}']
        static_avg_y_coord = _convert_to_np_array(static_patch['segs'][0]['avg_y_coords'][-common_pow_2:])
        static_act_y_coord = _convert_to_np_array(static_patch['segs'][0]['act_y_coords'][-common_pow_2:])
        
        movable_avg_y_coord = _convert_to_np_array(center_seg_avg[-common_pow_2:])
        movable_act_y_coord = _convert_to_np_array(center_seg_act[-common_pow_2:])
        
        avg_diff_y_coord = static_avg_y_coord - movable_avg_y_coord
        dy_coord = (static_act_y_coord - movable_act_y_coord) - avg_diff_y_coord
        
        dy_coords.append(dy_coord)
    
    # return self.signal_analyzer.calc_avg_signals(dy_coords)
    dy_coord_avg = _convert_to_np_array(dy_coords[0])
    for i in range(1, len(dy_coords)):
        dy_coord_avg += dy_coords[i]
    dy_coord_avg /= float(len(dy_coords))
    
    dy_fft_of_avg = _calc_fft(dy_coord_avg)
    
    s2n = _signal_to_noise(dy_fft_of_avg)
    print('Signal to noise: %s' % s2n)
    
    #def calc_avg_signals(self, ys: list[np.ndarray]) -> list[Signal]:
    #    yf_avg = self.calc_avg_fft(ys)
    #    peaks = self.find_peaks(yf_avg, height=SIGNAL_STRENGTH_THRESHOLD)
    #    return self.calc_signals(ys, yf_avg, peaks)
    SIGNAL_STRENGTH_THRESHOLD = 0.2
    peaks, _ = sp.signal.find_peaks(dy_fft_of_avg, height=SIGNAL_STRENGTH_THRESHOLD)
    print('peaks found: %s' % peaks)
    
    # Aug 8 2024 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LEFT OFF HERE
    # movable_patch.analyze_new_signals(
    signals = [ ]
    fft_size = len(dy_coords[0])
    fps = 30
    decimation = 6
    bpm_fft_precisions = {16: 18.75, 32: 9.375, 64: 4.6875, 128: 2.34375, 256: 1.171875, 512: 0.5859375}
    fft_lookup = {16: FftLevel.LEVEL_1, 32: FftLevel.LEVEL_2, 64: FftLevel.LEVEL_3, 128: FftLevel.LEVEL_4, 256: FftLevel.LEVEL_5, 512: FftLevel.LEVEL_6}
    
    print('fft size: %s' % fft_size)

    for fft_idx in peaks:
        signal = Signal()
        signal.strength = dy_fft_of_avg[fft_idx]
        signal.bpm_est = (fft_idx / (fft_size / (fps / decimation))) * 60
        signal.fft_size = fft_size
        signal.bpm_precision = bpm_fft_precisions[fft_size]
        signal.fft_level = fft_lookup[signal.fft_size]
        signal.decimation = decimation
        signals.append(signal)
    
    #for signal in signals:
    #    print('- %s' % signal)
    #BPM precisions: {16: 18.75, 32: 9.375, 64: 4.6875, 128: 2.34375, 256: 1.171875, 512: 0.5859375}
    #def calc_signals(self, ys: list[np.ndarray], yf_avg: np.ndarray, peaks: np.ndarray) -> list[Signal]:
    #    signals = [ ]
    #    fft_size = len(ys[0])
    #    for fft_idx in peaks:
    #        signal = Signal()
    #        signal.strength = yf_avg[fft_idx]
    #        signal.bpm_est = (fft_idx / (fft_size / (self.fps / self.decimation))) * 60
    #        signal.fft_size = fft_size
    #        signal.bpm_precision = self.bpm_fft_precisions[fft_size]
    #        signal.fft_level = self.fft_lookup[signal.fft_size]
    #        signal.decimation = self.decimation
    #        signals.append(signal)
    #    return signals
    
    if _can_do_time_domain_analysis(signals):
        print('Can do time domain analysis')
    else:
        print('Cannot do time domain analysis')
    
    peaks_with_no_peak_dist = [ ]
    for signal in signals:
        print('- Signal: %s' % signal)
        fft_index, min_bpm = _get_fft_index(signal)
        print('  fft index: %s, min bpm: %s' % (fft_index, min_bpm))
        
        fft_index_threshold = fft_index + 1
        
        # BPM   min_bpm * fft_index
        # BPS   BPM / 60.0
        peak_dist = int(60 / (fft_index_threshold * min_bpm) * decimation)
        print('  peak dist: %s' % peak_dist)
        
        peaks, _ = sp.signal.find_peaks(dy_coord_avg)
        peaks_with_no_peak_dist = peaks
        print('  peaks with no peak dist: %s, len(peaks): %s' % (peaks, len(peaks)))
        if peak_dist > 0:
            peaks, _ = sp.signal.find_peaks(
                dy_coord_avg,
                distance=peak_dist,
                height=SIGNAL_STRENGTH_THRESHOLD)
        else:
            print('  --- peak dist is zero')
        print('  peaks: %s, len(peaks): %s' % (peaks, len(peaks)))
        
        '''
        
        Left off here
            >>> inspect_patch_v2(32, 19)
            
            For Blaze data
            >>> inspect_patch_v2(32, 133)
            
                Seeing 3 peaks, 2nd one is super low relative to the others ...
                Would I make another check where the peaks should have a relatively close height?
                
                
            Blaze data looks ok, but no wthis doesn't work
            with non-blaze data
            
                >>> inspect_patch_v2(120, 19)
                Tried larger fft size
                I think some of the peaks are not found
        '''
        
        if len(peaks) >= 2:
            avg_peaks = [ ]
            for i in range(1, len(peaks)):
                avg_peaks.append(peaks[i] - peaks[i - 1])
            avg_peak_dist = np.average(avg_peaks)
            bpm = _convert_dist_to_bpm(avg_peak_dist)
            
            print('  avg peak dist: %s, bpm: %s' % (avg_peak_dist, bpm))
            print('  bpm: %s, min_bpm: %s, fft_index: %s' % (bpm, min_bpm, fft_index))
            if _bpm_in_range(bpm, min_bpm, fft_index):
                print('  BPM is in range')
            else:
                print('  BPM is out of range')
    
    # --- graph stuffs --------------------------------------------------------
    fig, axs = plt.subplots(nrows=3, ncols=1)
    
    x = np.arange(len(center_seg_avg))
    axs[0].plot(x, center_seg_avg)
    axs[0].plot(x, center_seg_act)
    
    for dy_coord in dy_coords:
        axs[1].plot(np.arange(len(dy_coord)), dy_coord)
    axs[1].plot(np.arange(len(dy_coord_avg)), dy_coord_avg + 5)
    if len(peaks_with_no_peak_dist) > 0:
        axs[1].scatter(peaks_with_no_peak_dist, np.zeros(len(peaks_with_no_peak_dist)))
    
    axs[2].plot(np.arange(len(dy_fft_of_avg)), dy_fft_of_avg)
    
    plt.show()
    
    #print(
    #fig, axs = plt.subplots(nrows=2, ncols=1)
    #axs[0].plot(cen, dy_coords_center_avg[peaks], "o", color='green')

def _convert_dist_to_bpm(avg_peak_dist):
    decimation = 6
    fps = 30
    return 60 / (avg_peak_dist * (decimation / fps))

def _calc_min_bpm(fft):
    fps = 30
    decimation = 6
    return (1 / (fft / (fps / decimation))) * 60

def _get_fft_index(signal: Signal):
    min_bpm = _calc_min_bpm(signal.fft_size)
    return int(signal.bpm_est / min_bpm), min_bpm

def _can_do_time_domain_analysis(signals):
    if not signals:
        print('No signals to do time domain analysis')
        return False
    
    if not signals:
        raise RuntimeError('Cannot check if time domain analysis can be '
            'performed, expecting signals to not be empty')
    
    min_bpm = 99999
    for signal in signals:
        if signal.bpm_est < min_bpm:
            min_bpm = signal.bpm_est
    
    print('min bpm: %s' % (min_bpm))
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
    
    print('time needed: %s, fft seconds: %s' % (time_needed, fft_seconds))
    
    return time_needed <= fft_seconds

def _bpm_in_range(bpm, min_bpm, fft_index):
    ''' Is bpm in the range described above like [fft-1 fft+1] '''
    min_fft_index = (fft_index - 1) if fft_index > 1 else .5
    max_fft_index = fft_index + 1
    print('is bpm in range? - min: %s, bpm: %s, max: %s' % (
        (min_fft_index * min_bpm),
        bpm,
        (max_fft_index * min_bpm)))
    return (min_fft_index * min_bpm) <= bpm <= (max_fft_index * min_bpm)

def inspect_patch(step_idx, patch_id):
    ''' Interesting patch ids:
    - Good 19
    - Bad  17
    '''
    global analyzer
    print('Num of steps: %s' % len(analyzer.patch_steps))
    step = analyzer.patch_steps[step_idx]
    
    static_patches = step['static_patches']
    movable_patches = step['movable_patches']
    
    analyzer.load_frame(int(step['frame_num']))
    
    try:
        movable_patch = movable_patches[str(patch_id)]
    except KeyError as err:
        print('Key error: %s' % err)
        print('movable patches len:  %s' % len(movable_patches))
        print('movable patches keys: %s' % movable_patches.keys())
        return
    
    try:
        print('movable_patch keys: %s' % movable_patch.keys())
        center_point = movable_patch['points'][0]
    except KeyError as err:
        #print('movable patch: %s' % movable_patch)
        print('movable patch keys: %s' % movable_patch.keys())
        print('segs len: %s' % len(movable_patch['segs']))
        return
    
    try:
        print('Point to render: %s, %s' % (center_point[0], center_point[1]))
        analyzer.draw_point(patch_id, center_point[0], center_point[1])
        analyzer.render_frame()
    except KeyError as err:
        print('Error ? %s' % err)
        print('center_point: %s' % center_point)
        return
    
    center_seg = movable_patch['segs'][0]
    center_seg_avg = center_seg['avg_y_coords']
    center_seg_act = center_seg['act_y_coords']
    # Keys: avg_y_coords, act_y_coords
    print('point count: %s' % len(center_seg_avg))
    
    movable_largest_pow_2 = \
            utils.get_largest_pow_of_2(len(center_seg_avg))
    print('Largest pow 2: %s' % movable_largest_pow_2)
    
    sp_pow_2_infos = [ ]
    for patch_id, static_patch in static_patches.items():
        coords_len = len(static_patch['segs'][0]['avg_y_coords'])
        sp_pow_2 = utils.get_largest_pow_of_2(coords_len)
        
        # self.signal_analyzer.get_min_fft_size()
        # Decimation is 6, min fft size is 16
        if sp_pow_2 < 16:
            continue
        sp_pow_2_infos.append(PatchPow2Info(int(patch_id), sp_pow_2))
    
    best_sp_idxs = patch_pow_2_info.find_best_infos(
        movable_largest_pow_2,
        sp_pow_2_infos)
    print('>>> best sp idx: %s, movable largest pow of 2: %s' % (best_sp_idxs, movable_largest_pow_2))
    for i in best_sp_idxs:
        print('- [%s] %s' % (i, sp_pow_2_infos[i]))
    
    best_static_patch = static_patches[f'{best_sp_idxs[0]}']
    best_static_patch_coords_len = len(best_static_patch['segs'][0]['avg_y_coords'])
    common_pow_2 = min(
        movable_largest_pow_2,
        utils.get_largest_pow_of_2(best_static_patch_coords_len))
    print('Common pow 2: %s' % common_pow_2)
    
    # def _analyze_movable_seg(self,
    #    movable_seg: Segment,
    #    static_patches: list[StaticPatch],
    #    best_sp_idxs: list[int],
    #    common_pow_2: int) -> list[Signal]:
    dy_coords = [ ]
        
    for best_sp_idx in best_sp_idxs:
        static_patch = static_patches[f'{best_sp_idxs[best_sp_idx]}']
        static_avg_y_coord = _convert_to_np_array(static_patch['segs'][0]['avg_y_coords'][-common_pow_2:])
        static_act_y_coord = _convert_to_np_array(static_patch['segs'][0]['act_y_coords'][-common_pow_2:])
        
        movable_avg_y_coord = _convert_to_np_array(center_seg_avg[-common_pow_2:])
        movable_act_y_coord = _convert_to_np_array(center_seg_act[-common_pow_2:])
        
        avg_diff_y_coord = static_avg_y_coord - movable_avg_y_coord
        dy_coord = (static_act_y_coord - movable_act_y_coord) - avg_diff_y_coord
        
        dy_coords.append(dy_coord)
    
    dy_avg_fft = _calc_avg_fft(dy_coords)
    
    dy_coord_avg = _convert_to_np_array(dy_coords[0])
    for i in range(1, len(dy_coords)):
        dy_coord_avg += dy_coords[i]
    dy_coord_avg /= float(len(dy_coords))
    
    dy_fft_of_avg = _calc_fft(dy_coord_avg)
    
    s2n = _signal_to_noise(dy_avg_fft)
    print('Signal to noise: %s' % s2n)
    
    #def calc_avg_signals(self, ys: list[np.ndarray]) -> list[Signal]:
    #    yf_avg = self.calc_avg_fft(ys)
    #    peaks = self.find_peaks(yf_avg, height=SIGNAL_STRENGTH_THRESHOLD)
    #    return self.calc_signals(ys, yf_avg, peaks)
    SIGNAL_STRENGTH_THRESHOLD = 0.5
    peaks, _ = sp.signal.find_peaks(dy_avg_fft, height=SIGNAL_STRENGTH_THRESHOLD)
    print('peaks found: %s' % peaks)
    
    signals = [ ]
    fft_size = len(dy_coords[0])
    fps = 30
    decimation = 6
    bpm_fft_precisions = {16: 18.75, 32: 9.375, 64: 4.6875, 128: 2.34375, 256: 1.171875, 512: 0.5859375}
    fft_lookup = {16: FftLevel.LEVEL_1, 32: FftLevel.LEVEL_2, 64: FftLevel.LEVEL_3, 128: FftLevel.LEVEL_4, 256: FftLevel.LEVEL_5, 512: FftLevel.LEVEL_6}
    
    print('fft size: %s' % fft_size)

    for fft_idx in peaks:
        signal = Signal()
        signal.strength = dy_avg_fft[fft_idx]
        signal.bpm_est = (fft_idx / (fft_size / (fps / decimation))) * 60
        signal.fft_size = fft_size
        signal.bpm_precision = bpm_fft_precisions[fft_size]
        signal.fft_level = fft_lookup[signal.fft_size]
        signal.decimation = decimation
        signals.append(signal)
    
    for signal in signals:
        print('- %s' % signal)
    #BPM precisions: {16: 18.75, 32: 9.375, 64: 4.6875, 128: 2.34375, 256: 1.171875, 512: 0.5859375}
    #def calc_signals(self, ys: list[np.ndarray], yf_avg: np.ndarray, peaks: np.ndarray) -> list[Signal]:
    #    signals = [ ]
    #    fft_size = len(ys[0])
    #    for fft_idx in peaks:
    #        signal = Signal()
    #        signal.strength = yf_avg[fft_idx]
    #        signal.bpm_est = (fft_idx / (fft_size / (self.fps / self.decimation))) * 60
    #        signal.fft_size = fft_size
    #        signal.bpm_precision = self.bpm_fft_precisions[fft_size]
    #        signal.fft_level = self.fft_lookup[signal.fft_size]
    #        signal.decimation = self.decimation
    #        signals.append(signal)
    #    return signals
    
    # --- graph stuffs --------------------------------------------------------
    fig, axs = plt.subplots(nrows=3, ncols=1)
    
    x = np.arange(len(center_seg_avg))
    axs[0].plot(x, center_seg_avg)
    axs[0].plot(x, center_seg_act)
    
    for dy_coord in dy_coords:
        axs[1].plot(np.arange(len(dy_coord)), dy_coord)
    axs[1].plot(np.arange(len(dy_coord_avg)), dy_coord_avg + 5)
    
    axs[2].plot(np.arange(len(dy_avg_fft)), dy_avg_fft)
    axs[2].plot(np.arange(len(dy_fft_of_avg)), dy_fft_of_avg)
    
    plt.show()
    
    #print(
    #fig, axs = plt.subplots(nrows=2, ncols=1)
    #axs[0].plot(cen, dy_coords_center_avg[peaks], "o", color='green')

def _calc_avg_fft(ys: list[np.ndarray]) -> np.ndarray:
    yf_avg = _calc_fft(ys[0])
    for i in range(1, len(ys)):
        yf_avg += _calc_fft(ys[i])
    return yf_avg

def _calc_fft(y: np.ndarray) -> np.ndarray:
    yf = sp.fft.fft(y)
    yf = 2.0 / len(y) * np.abs(yf[:len(y) // 2])
    return yf

def _convert_to_np_array(vals: list):
    return np.asarray(vals, dtype=float)

def _signal_to_noise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def inspect_step(step_idx):
    global analyzer
    print('Num of steps:        %s' % len(analyzer.patch_steps))
    
    if step_idx >= len(analyzer.patch_steps):
        print('Step index out of bounds')
        return

    step = analyzer.patch_steps[step_idx]
    
    print('Step keys: %s' % step.keys())
    print('- Frame #:    %s' % step['frame_num'])
    print('')
    
    analyzer.load_frame(int(step['frame_num']))
    
    print('Static patches: %s' % step['static_patches'].keys())
    for static_id, static_patch in step['static_patches'].items():
        print('- id: %s, static patch: %s' % (static_id, static_patch.keys()))
        print('- Num segs: %s' % len(static_patch['segs']))
        for seg in static_patch['segs']:
            print('- Seg: %s' % seg.keys())
            for key, val in seg.items():
                print('  - key: %s, val: %s' % (key, val))
        print('- Points: %s' % static_patch['points'])
        break
    print('')
    
    print('Movable patches - Num patches %s' % len(step['movable_patches']))
    for patch_id, patch in step['movable_patches'].items():
        #print('patch - seg count: %s, point count: %s' % (len(patch['segs']), len(patch['points'])))
        center_point = patch['points'][0]
        analyzer.draw_point(patch_id, center_point[0], center_point[1])
    
    analyzer.render_frame()
    
    '''
    - There's a bunch of movable patches, about 50+. I want to load the frame, draw the static and movable patches, number them. So I can start selecting patches based on patch number and start examining how they are false positives or not getting the right answer or not being filtered out based on the state machine.
    '''

def examine_fft(step_idx, min_fft_size):
    global analyzer
    print('Num of steps:        %s' % len(analyzer.patch_steps))
    
    if step_idx >= len(analyzer.patch_steps):
        print('Step index out of bounds')
        return

    step = analyzer.patch_steps[step_idx]
    
    static_patches = get_patches(step['static_patches'], min_fft_size)
    movable_patches = get_patches(step['movable_patches'], min_fft_size)
    
    print('Num static patches:  %s' % len(static_patches))
    print('Num movable patches: %s' % len(movable_patches))
    
    if len(static_patches) == 0 or len(movable_patches) == 0:
        print('Zero static and/or movable patches, cannot examine')
        return
    
    x = np.arange(min_fft_size)
    alpha = .35
    
    fig, axs = plt.subplots(nrows=3, ncols=3)
    fig1, axs1 = plt.subplots(nrows=3, ncols=1)
    
    movable_patch_keys = list(movable_patches.keys())
    print('movable_patch_keys', movable_patch_keys)
    
    movable_patches = {
        '0': movable_patches[movable_patch_keys[0]],
        '1': movable_patches[movable_patch_keys[len(movable_patch_keys) // 2]],
        '2': movable_patches[movable_patch_keys[-1]],
    }
    
    for movable_patch_id, movable_patch in movable_patches.items():
        movable_patch_id = int(movable_patch_id)
        movable_segs = [SimpleSeg(s) for s in movable_patch['segs']]
        
        #fig, axs = plt.subplots(nrows=1, ncols=2)
        #axs[1].set_ylim([0, 2])
        
        dy_coords = [ ]
        dy_coords_center_avg = None
        dy_coords_center_avg_count = 0
        
        for movable_seg_idx, movable_seg in enumerate(movable_segs):
            #if movable_seg_idx > 0:
            #    break
            
            movable_avg_y_coords, movable_act_y_coords = movable_seg.get_latest(min_fft_size)
            
            for static_patch_id, static_patch in static_patches.items():
                static_seg = SimpleSeg(static_patch['segs'][0])
                static_avg_y_coords, static_act_y_coords = static_seg.get_latest(min_fft_size)
                
                avg_diff_y_coord = static_avg_y_coords - movable_avg_y_coords
                dy_coord = (static_act_y_coords - movable_act_y_coords) - avg_diff_y_coord
                fft_coord = calc_fft(dy_coord)
                
                dy_coords.append(dy_coord)
                
                axs[movable_patch_id][0].plot(x, dy_coord, alpha=alpha)
                axs[movable_patch_id][1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha)
                
                if movable_seg_idx == 0:
                    if dy_coords_center_avg is None:
                        dy_coords_center_avg = dy_coord
                    else:
                        dy_coords_center_avg += dy_coord
                    dy_coords_center_avg_count += 1
        
        # Blaze2 data use distance = 14
        # Wagner data use distance = 3
        peaks, _ = sp.signal.find_peaks(dy_coords_center_avg / dy_coords_center_avg_count, distance=14)
        #peaks, _, _ = find(dy_coords_center_avg / dy_coords_center_avg_count, 5)
        if len(peaks) > 0:
            print('peaks 1', peaks)
            print('dy_coords_center_avg', dy_coords_center_avg[peaks])
            print('')
        axs1[movable_patch_id].plot(peaks, dy_coords_center_avg[peaks], "o", color='green')
        axs1[movable_patch_id].plot(x, dy_coords_center_avg, alpha=alpha)
        
        fft_coords_avg = calc_avg_fft(dy_coords)
        peaks, props = sp.signal.find_peaks(fft_coords_avg, height=.5) #, prominence=.5)
        axs[movable_patch_id][2].plot(peaks, fft_coords_avg[peaks], "o", color='green')
        axs[movable_patch_id][2].plot(np.arange(len(fft_coords_avg)), fft_coords_avg, alpha=1.0)
        print('peaks 2', peaks, 'groups', props)
    
    plt.show()

def get_patches(patch_dict, min_fft_size):
    patches = { }
    for group_id, patch in patch_dict.items():
        for seg in patch['segs']:
            if len(seg['avg_y_coords']) >= min_fft_size:
                patches[group_id] = patch
    return patches

def calc_fft(y: np.ndarray) -> np.ndarray:
    yf = sp.fft.fft(y)
    yf = 2.0 / len(y) * np.abs(yf[:len(y) // 2])
    return yf

def calc_avg_fft(ys: list[np.ndarray]) -> np.ndarray:
    yf_avg = calc_fft(ys[0])
    for i in range(1, len(ys)):
        yf_avg += calc_fft(ys[i])
    return yf_avg / len(ys)

def get_latest_buf(buf: np.ndarray, count: int) -> np.ndarray:
    if count > len(buf):
        raise Exception('Cannot get latest buffer - count: %s, size: %s' % (
            count, len(buf)))
    return buf[len(buf) - count:len(buf)]

class SimpleSeg:
    def __init__(self, seg):
        self.avg_y_coords = np.asarray(seg['avg_y_coords'], dtype=float)
        self.act_y_coords = np.asarray(seg['act_y_coords'], dtype=float)
        
        #self.avg_y_coords = self.avg_y_coords - self.avg_y_coords[0]
        #self.act_y_coords = self.act_y_coords - self.act_y_coords[0]
    
    def get_latest(self, count):
        avg_y_coords = get_latest_buf(self.avg_y_coords, count)
        act_y_coords = get_latest_buf(self.act_y_coords, count)
        return avg_y_coords, act_y_coords
    
    def get_size(self):
        return len(self.avg_y_coords)






###############################################################################
# From bpm_peak_finder.py
###############################################################################
# PXL_20230824_035440033, I see a distance at about 18, setting to 17 which would give a bpm of about 106
# 60/(17/30) = 105.88235294117648
def find(y, distance=17):
    '''
    @param y Points to search through
    @param distance Distance used in find peaks
    @return peaks, prominences, peaks std
    '''
    min_std = 99999999
    std_prev = 0
    std_list = [ ]
    std_state = 0
    std = min_std
    last_known_peaks_prom = 0
    
    found = False
    
    for prominence in range(2, 20):
        peaks, proms, std = _calc_peaks(y, prominence, distance)
        
        if len(peaks) <= 1:
            continue
        
        last_known_peaks_prom = prominence
        
        if std == 0:
            std_list.append(0)
            continue
        else:
            std_list.append(std)
        
        # Expecting std to:
        # - start low with low prominence
        # - std increase, then std peaks
        # - std decrease again, once hit bottom, that's the prominence to use
        if std_state == 0:
            # Skip this round, waiting for std_prev to be init'd
            std_state = 1
        elif std_state == 1:
            if std > std_prev:
                # On the up tick
                std_state = 1
            else:
                std_state = 2
        elif std_state == 2:
            if std < std_prev:
                std_state = 2
            else:
                std_state = 3
        if std_state == 3:
            found = True
            break
        
        std_prev = std
        
        if std < min_std and std > 0:
            min_std = std

    if not found:
        if last_known_peaks_prom > 0:
            peaks, proms, std = _calc_peaks(y, last_known_peaks_prom, distance)
        
        if len(peaks) <= 1:
            peaks = [ ]
            proms = [ ]
            std = 99999

    return peaks, proms, std

def _calc_peaks(y, prominence, distance):
    peaks, props = sp.signal.find_peaks(
        y,
        prominence=prominence,
        distance=distance)
    
    if len(peaks) > 1:
        proms = props['prominences']
        
        dx_peaks = [ ]
        for j in range(1, len(peaks)):
            dx_peaks.append(peaks[j] - peaks[j - 1])
        
        std = np.std(dx_peaks)
    else:
        peaks = [ ]
        proms = [ ]
        std = 99999

    return peaks, proms, std

def find_debug(y, distance=17):
    peaks, props = sp.signal.find_peaks(
        y,
        prominence=7,
        distance=distance)
    
    proms = props['prominences']
    
    dx_peaks = [ ]
    for j in range(1, len(peaks)):
        dx_peaks.append(peaks[j] - peaks[j - 1])
    
    std = np.std(dx_peaks)
    
    return peaks, proms, std
###############################################################################











