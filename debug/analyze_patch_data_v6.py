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
from pet_breathy.fft_overlapper import FftOverlapper
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

exec(open('./debug/analyze_patch_data_v6.py').read())
load_analyzer_v2()

'''

g_use_blaze = True

def load_analyzer_v2():
    if not g_use_blaze:
        fn = './debug/data/PXL_20230825_040038487.json'
    else:
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

class BreathySim:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        print('Num of steps: %s' % len(self.analyzer.patch_steps))
        
        # Key:   patch_id
        # Value: [ fft ]
        self.patch_stats = { }

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            pass

    def _run(self):
        self.num_overlaps = self.analyzer.video_info.fps // self.analyzer.decimation
        print('Num overlaps: %s' % self.num_overlaps)
        self.overlapper = FftOverlapper(self.num_overlaps)
        
        self.min_fft_len = 16
        
        # Key:      fft size
        # Val:      list[ spectra ]
        self.spectra_dict = { }
        
        self.orig_patch_ids = [ ]
        if g_use_blaze:
            self.orig_patch_ids.append(38) # Blaze2, good point
            #self.orig_patch_ids.append(70) # Blaze2, good point
            #self.orig_patch_ids.append(85) # Blaze2, bad point
            #self.orig_patch_ids.append(6) # Blaze2, bad point
        else:
            self.orig_patch_ids.append(66) # Mr Pups, good point
            self.orig_patch_ids.append(13) # Mr pups, bad point
        
        for self.orig_patch_id in self.orig_patch_ids:
            for self.step_idx in range(1, len(self.analyzer.patch_steps)):
                self.step = self.analyzer.patch_steps[self.step_idx]
                self._analyze_step()
        
        self._plot_spectra()

    def _analyze_step(self):
        self.static_patches = self.step['static_patches']
        self.movable_patches = self.step['movable_patches']
        
        self.patch_id = self.orig_patch_id
        
        self.movable_patch = self.movable_patches[str(self.patch_id)]
        
        try:
            self.patch_ffts = self.patch_stats[self.patch_id][0]
            self.dy_coord_avgs = self.patch_stats[self.patch_id][1]
        except KeyError:
            self.patch_ffts = [ ]
            self.dy_coord_avgs = [ ]
            self.patch_stats[self.patch_id] = [ self.patch_ffts, self.dy_coord_avgs ]
        
        self._analyze_patch()
    
    def _analyze_patch(self):
        self.seg = self.movable_patch['segs'][0]
        self._analyze_segment()

    def _analyze_segment(self):
        seg_avg = self.seg['avg_y_coords']
        seg_act = self.seg['act_y_coords']
        
        if len(seg_act) < self.min_fft_len:
            return
        
        common_pow_2 = utils.get_largest_pow_of_2(len(seg_act))
        
        static_do_not_compare = [ ]
        for self.patch_id, static_patch in self.static_patches.items():
            coords_len = len(static_patch['segs'][0]['avg_y_coords'])
            if coords_len != len(seg_avg):
                static_common_pow_2 = utils.get_largest_pow_of_2(coords_len)
                if static_common_pow_2 < common_pow_2:
                    #common_pow_2 = min(common_pow_2, coords_len)
                    #raise RuntimeError('static patch length doesn\'t match seg')
                    static_do_not_compare.append(self.patch_id)
                    print(self.patch_id, coords_len, common_pow_2)
        
        if len(static_do_not_compare) > 2:
            raise RuntimeError('Not enough static points to do comparison')
        
        dy_coords = [ ]
        
        for self.patch_id, static_patch in self.static_patches.items():
            if self.patch_id in static_do_not_compare:
                continue
            #static_patch = self.static_patches[f'{best_sp_idxs[best_sp_idx]}']
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
        
        self.overlapper.add_vals(dy_coord_avg)
        
        try:
            spectras = self.spectra_dict[common_pow_2]
        except KeyError:
            spectras = [ ]
            self.spectra_dict[common_pow_2] = spectras
        
        #if self.overlapper.get_spectras_count() >= self.num_overlaps:
        #    spectras.append(self.overlapper.get_spectra())
        spectras.append([
            self.overlapper.get_spectra(),
            self.overlapper.get_last_fft(),
            #self.overlapper.get_last_raw_fft(),
            ])
        
        #self._plot_step()

    def _plot_step(self):
        spectra = self.overlapper.get_spectra()
        
        fig, axs = plt.subplots(nrows=1, ncols=1)
        axs.plot(np.arange(len(spectra)), spectra)
        
        plt.show()

    def _plot_spectra(self):
        num_keys = len(self.spectra_dict.keys())
        fig, axs = plt.subplots(nrows=num_keys, ncols=2)
        axs_idx = 0
        
        for fft_len, spectras in self.spectra_dict.items():
            print('- FFT len: %s, num spectras: %s' % (fft_len, len(spectras)))
            for i, items in enumerate(spectras):
                spectra = items[0]
                last_fft = items[1]
                #last_raw_fft = items[2]
                
                spectra_peaks = find_peaks_with_height(spectra)
                last_fft_peaks = find_peaks_with_height(last_fft)
                print('-', spectra_peaks)
                #print(last_fft_peaks)
                print(' ', spectra[spectra_peaks])
                
                ''' Could use the top 3
                Sort the peaks with the indexes
                
                >>> l = [1,2,3]
                >>> m = [99,2,1]
                >>> [(x,y) for x,y in sorted(zip(l,m), key=lambda pair: pair[1], reverse=True)]
                [(1, 99), (2, 2), (3, 1)]
                
                where l corresponds to peak indexes
                where m corresponds to peak values
                '''
                
                axs[axs_idx][0].plot(np.arange(len(spectra)), spectra)
                axs[axs_idx][1].plot(np.arange(len(last_fft)), last_fft)
                #axs[axs_idx][1].plot(np.arange(len(last_raw_fft)), last_raw_fft)
                
                axs[axs_idx][0].scatter(spectra_peaks, spectra[spectra_peaks])
                # axs[1].scatter(peaks_with_no_peak_dist, np.zeros(len(peaks_with_no_peak_dist)))
            axs_idx += 1
        
        plt.show()

def find_peaks_no_height(vals):
    peaks, _ = sp.signal.find_peaks(vals)
    return peaks

def find_peaks_with_height(vals):
    ''' Given a few measurements on blaze/pups data, .005 seems reasonable in filtering points
    '''
    peaks, _ = sp.signal.find_peaks(vals, height=0.005)
    return peaks


def _convert_to_np_array(vals: list):
    return np.asarray(vals, dtype=float)

def _calc_fft(y: np.ndarray) -> np.ndarray:
    yf = sp.fft.fft(y)
    yf = 2.0 / len(y) * np.abs(yf[:len(y) // 2])
    return yf





