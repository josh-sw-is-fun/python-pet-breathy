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

exec(open('./debug/analyze_patch_data_v3.py').read())
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

def inspect_patch_v3_frame(frame_num, patch_id, fft_size_override=-1):
    found_step_idx = False
    for step_idx, step in enumerate(analyzer.patch_steps):
        if int(step['frame_num']) == frame_num:
            print('Found step idx: %s' % step_idx)
            inspect_patch_v3(step_idx, patch_id, fft_size_override)
            found_step_idx = True
            break
    print('Found step idx: %s' % found_step_idx)

def inspect_patch_v3(step_idx, patch_id, fft_size_override=-1):
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
    
    if fft_size_override != -1:
        movable_largest_pow_2 = fft_size_override
    else:
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
    
    for sp_pow_2_info in sp_pow_2_infos:
        print('- %s' % sp_pow_2_info)
    
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
    
    print('dy_coord_avg:', len(dy_coord_avg), dy_coord_avg)
    print('dy_fft_of_avg:', len(dy_fft_of_avg))
    for i in range(len(dy_fft_of_avg)):
        print('- fft[%3d] %s' % (i, dy_fft_of_avg[i]))
    
    #def calc_avg_signals(self, ys: list[np.ndarray]) -> list[Signal]:
    #    yf_avg = self.calc_avg_fft(ys)
    #    peaks = self.find_peaks(yf_avg, height=SIGNAL_STRENGTH_THRESHOLD)
    #    return self.calc_signals(ys, yf_avg, peaks)
    SIGNAL_STRENGTH_THRESHOLD = 0.2
    peaks, x = sp.signal.find_peaks(
        dy_fft_of_avg,
        height=SIGNAL_STRENGTH_THRESHOLD)
    print('peaks found: %s' % peaks)
    print('--> x', x)
    
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
    
    if _can_do_time_domain_analysis(signals):
        print('Can do time domain analysis')
    else:
        print('Cannot do time domain analysis')
    
    peaks_with_no_peak_dist = np.arange(0)
    peaks_with_peak_dist = np.arange(0)
    bpms = [ ]
    res = None
    for signal in signals:
        print('--------------------------------------------')
        print('- Signal: %s' % signal)
        fft_index, min_bpm = _get_fft_index(signal)
        print('  fft index: %s, min bpm: %s' % (fft_index, min_bpm))
        
        energy_in_prev_fft = False
        energy_in_next_fft = False
        
        prev_fft_index = max(fft_index - 1, 1)
        next_fft_index = min(fft_index + 1, len(dy_fft_of_avg) - 1)
        if prev_fft_index != fft_index:
            frac = dy_fft_of_avg[prev_fft_index] / dy_fft_of_avg[fft_index]
            if dy_fft_of_avg[prev_fft_index] * .5 >= SIGNAL_STRENGTH_THRESHOLD and \
                    frac >= 0.75:
                print(' !!!! Additional energy in prev fft bucket, fft: %s, prev fft: %s, frac: %s !!!!' % (dy_fft_of_avg[fft_index], dy_fft_of_avg[prev_fft_index], frac))
                energy_in_prev_fft = True
        if next_fft_index != fft_index:
            frac = dy_fft_of_avg[next_fft_index] / dy_fft_of_avg[fft_index]
            if dy_fft_of_avg[next_fft_index] * .5 >= SIGNAL_STRENGTH_THRESHOLD and \
                    frac >= 0.75:
                print(' !!!! Additional energy in next fft bucket, fft: %s, next fft: %s, frac: %s !!!!' % (dy_fft_of_avg[fft_index], dy_fft_of_avg[next_fft_index], frac))
                energy_in_next_fft = True
        #min_fft_index = max(fft_index - 1, 1)
        #max_fft_index = min(fft_index + 1 + 1, len(dy_fft_of_avg) - 1)
        #for temp_fft_index in range(min_fft_index, max_fft_index):
        #    print('  - [%d] fft: %s' % (temp_fft_index, dy_fft_of_avg[temp_fft_index]))
        
        fft_index_threshold = fft_index + 1
        
        # BPM   min_bpm * fft_index
        # BPS   BPM / 60.0
        peak_dist = int(60 / (fft_index_threshold  * min_bpm) * (fps / decimation))
        print('  peak dist: %s' % peak_dist)
        
        peaks_with_no_peak_dist, _ = sp.signal.find_peaks(dy_coord_avg)
        print('  peaks with no peak dist: %s, len(peaks): %s' % (peaks_with_no_peak_dist, len(peaks_with_no_peak_dist)))
        if peak_dist > 0:
            peaks_with_peak_dist, _ = sp.signal.find_peaks(
                dy_coord_avg,
                distance=peak_dist,
                height=SIGNAL_STRENGTH_THRESHOLD)
        else:
            print('  --- peak dist is zero')
        print('  peaks with peak dist: %s, len(peaks): %s' % (peaks_with_peak_dist, len(peaks_with_peak_dist)))
        
        if len(peaks_with_peak_dist) >= 2:
            avg_peaks = [ ]
            for i in range(1, len(peaks_with_peak_dist)):
                avg_peaks.append(peaks_with_peak_dist[i] - peaks_with_peak_dist[i - 1])
            avg_peak_dist = np.average(avg_peaks)
            bpm = _convert_dist_to_bpm(avg_peak_dist)
            
            tt = np.linspace(0, len(dy_coord_avg), len(dy_coord_avg))
            tt2 = np.linspace(0, len(dy_coord_avg), len(dy_coord_avg)) # *len(dy_coord_avg))
            
            # guess bpm / ((fps / decimation) * 60) = guess freq
            #guess_freq = bpm / ((fps / decimation) * 60)
            guess_freq = signal.bpm_est / ((fps / decimation) * 60)
            print('  >>>>>> guess freq', guess_freq)
            res = fit_sin_with_guess(tt, dy_coord_avg, guess_freq)
            #print(res)
            #print('fit curve y-coords', res["fitfunc"](tt2))
            y0 = np.absolute(dy_coord_avg)
            y1 = np.absolute(res["fitfunc"](tt2))
            dist = np.sum(y1 - y0)
            
            found_bpm = False
            
            print('  y0', len(y0), 'y1', len(y1), 'dist', dist)
            
            print('  avg peak dist: %s, bpm: %s' % (avg_peak_dist, bpm))
            print('  bpm: %s, min_bpm: %s, fft_index: %s' % (bpm, min_bpm, fft_index))
            #if _bpm_in_range(bpm, min_bpm, fft_index) or signal.strength > 1.25:
            if _bpm_in_range(bpm, min_bpm, fft_index):
                print('  ==============> :) BPM is in range')
                bpms.append(bpm)
                found_bpm = True
                #bpms.append(signal.bpm_est)
                #break
            else:
                print('  BPM is out of range')
            
            if energy_in_prev_fft and not found_bpm:
                if _bpm_in_range(bpm, min_bpm, fft_index - 1):
                    print('  ==============> :) BPM of prev bucket is in range')
                    bpms.append(bpm)
                    found_bpm = True
                else:
                    print('  BPM of prev bucket is out of range')
            
            if energy_in_next_fft and not found_bpm:
                if _bpm_in_range(bpm, min_bpm, fft_index + 1):
                    print('  ==============> :) BPM of next bucket is in range')
                    bpms.append(bpm)
                    found_bpm = True
                else:
                    print('  BPM of next bucket is out of range')
    
    print('Num bpms found: %s, %s' % (len(bpms), bpms))
    
    
    tt = np.linspace(0, len(dy_coord_avg), len(dy_coord_avg))
    tt2 = np.linspace(0, len(dy_coord_avg), len(dy_coord_avg)) # *len(dy_coord_avg))
    #res = fit_sin(tt, dy_coord_avg)
    
    res = None
    if res is None and len(bpms) > 0:
        # guess bpm / ((fps / decimation) * 60) = guess freq
        
        min_dist = 999999.0
        best_bpm = min_dist
        best_guess_freq = 0
        for potential_bpm in bpms:
            print('   bpms centered on', bpms[0])
            min_bpm = int(potential_bpm - 3)
            max_bpm = int(potential_bpm + 3)
            for bpm in range(min_bpm, max_bpm):
                guess_freq = bpm / ((fps / decimation) * 60)
                #guess_freq = 0.230712890625
                #print('>>>>>> guess freq', guess_freq, 'bpm', bpm)
                res = fit_sin_with_guess(tt, dy_coord_avg, guess_freq)
                #print(res)
                #print('fit curve y-coords', res["fitfunc"](tt2))
                y0 = np.absolute(dy_coord_avg)
                y1 = np.absolute(res["fitfunc"](tt2))
                
                # This form is the one to use
                dist = np.absolute(np.sum(y1 - y0))
                
                # This form will not work since we do not get the sin canceling
                #dist = np.sum(np.absolute(y1 - y0))
                print('$$$ bpm', bpm, 'dist', dist) #, 'sum', np.sum(np.absolute(y1 - y0)))
                if dist < min_dist:
                    min_dist = dist
                    best_bpm = bpm
                    best_guess_freq = guess_freq
        print('---> min dist:', min_dist, 'best bpm', best_bpm)
        
        if best_guess_freq != 0:
            res = fit_sin_with_guess(tt, dy_coord_avg, best_guess_freq)
            #print(res)
            #print('fit curve y-coords', res["fitfunc"](tt2))
            y0 = np.absolute(dy_coord_avg)
            y1 = np.absolute(res["fitfunc"](tt2))
            dist = np.sum(y1 - y0)
            print('y0', len(y0), 'y1', len(y1), 'dist', dist)
        
        if False:
            guess_freq = bpms[0] / ((fps / decimation) * 60)
            #guess_freq = 0.230712890625
            print('>>>>>> guess freq', guess_freq, 'bpms', bpms)
            res = fit_sin_with_guess(tt, dy_coord_avg, guess_freq)
            #print(res)
            #print('fit curve y-coords', res["fitfunc"](tt2))
            y0 = np.absolute(dy_coord_avg)
            y1 = np.absolute(res["fitfunc"](tt2))
            dist = np.sum(y1 - y0)
            print('y0', len(y0), 'y1', len(y1), 'dist', dist)
    
    # --- graph stuffs --------------------------------------------------------
    fig, axs = plt.subplots(nrows=4, ncols=1)
    
    x = np.arange(len(center_seg_avg))
    axs[0].plot(x, center_seg_avg)
    axs[0].plot(x, center_seg_act)
    
    for dy_coord in dy_coords:
        axs[1].plot(np.arange(len(dy_coord)), dy_coord)
    axs[1].plot(np.arange(len(dy_coord_avg)), dy_coord_avg + 5)
    if len(peaks_with_no_peak_dist) > 0:
        axs[1].scatter(peaks_with_no_peak_dist, np.zeros(len(peaks_with_no_peak_dist)))
    if len(peaks_with_peak_dist) > 0:
        axs[1].scatter(peaks_with_peak_dist, np.zeros(len(peaks_with_peak_dist)) - 2)
    
    axs[2].plot(np.arange(len(dy_fft_of_avg)), dy_fft_of_avg)
    
    #axs[3].plot(np.arange(len(freq)), freq)
    #axs[3].plot(np.arange(len(fourier)), fourier)
    if res is not None:
        axs[3].plot(tt2, res["fitfunc"](tt2), "r-", label="y fit curve", linewidth=2)
    axs[3].plot(np.arange(len(dy_coord_avg)), dy_coord_avg)
    
    plt.show()
    
    #print(
    #fig, axs = plt.subplots(nrows=2, ncols=1)
    #axs[0].plot(cen, dy_coords_center_avg[peaks], "o", color='green')

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

def _convert_dist_to_bpm(avg_peak_dist):
    decimation = 6.0
    fps = 30.0
    return 60.0 / (avg_peak_dist * (decimation / fps))

def _calc_min_bpm(fft):
    fps = 30.0
    decimation = 6.0
    return (1.0 / (fft / (fps / decimation))) * 60.0

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

def _calc_fft(y: np.ndarray) -> np.ndarray:
    yf = sp.fft.fft(y)
    yf = 2.0 / len(y) * np.abs(yf[:len(y) // 2])
    return yf

def _convert_to_np_array(vals: list):
    return np.asarray(vals, dtype=float)

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    print('>>>>>>>>>> guess freq', guess_freq)
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = sp.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


# >>> 0.2119140625 * 5 * 60
# 63.57421875
#
# guess freq = 0.2119140625
# 5 is fps / decimation
# 60 is 60 seconds in a minute
# 63.574 is the guessed bpm
#
# guess bpm = guess freq * (fps / decimation) * 60
#
# guess bpm / ((fps / decimation) * 60) = guess freq

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











