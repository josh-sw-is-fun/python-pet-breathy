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
import scipy as sp
import cv2 as cv

'''
PYTHONPATH=$(pwd) python3

exec(open('./debug/analyze_patch_data_v5.py').read())
load_analyzer_v2()

'''

def load_analyzer_v2():
    fn = './debug/data/PXL_20230825_040038487.json'
    #fn = './debug/data/Blaze2.json'
    
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
        #for self.step_idx, self.step in enumerate(self.analyzer.patch_steps):
        #    self.frame_num = int(self.step['frame_num'])
        #    print('Frame #', self.frame_num)
        #    self._analyze_step()
        
        try:
            self._run()
        except KeyboardInterrupt:
            pass

    def _run(self):
        self.fft_len = 32
        self.min_seg_len = self.fft_len + self.fft_len // 2
        
        self.fft_win = np.hanning(self.fft_len) # // 2)
        
        for self.step_idx in range(1, len(self.analyzer.patch_steps)):
            self.prev_step = self.analyzer.patch_steps[self.step_idx - 1]
            self.step = self.analyzer.patch_steps[self.step_idx]
            self.prev_frame_num = int(self.prev_step['frame_num'])
            self.frame_num = int(self.step['frame_num'])
            #print('Prev frame: %s, frame: %s' % (self.prev_frame_num, self.frame_num))
            self._analyze_step()
        
        #self._plot_fft_avg_of_all()
        self._plot_fft_avg_of_slice()

    def _analyze_step(self):
        #self.analyzer.load_frame(int(self.step['frame_num']))
        
        self.static_patches = self.step['static_patches']
        self.movable_patches = self.step['movable_patches']
        
        #print('static patches: %s, movable patches: %s' % (len(self.static_patches), len(self.movable_patches)))
        #for self.patch_id, self.movable_patch in self.movable_patches.items():
        #    self.patch_id = int(self.patch_id)
        #    if self.patch_id == 70:
        #        self._analyze_patch()
        
        self.patch_id = 70 # Blaze2
        self.patch_id = 66 # Mr Pups, good point
        #self.patch_id = 13 # Mr pups, bad point
        
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
        
        if len(seg_act) < self.fft_len:
            return
        
        #print('- Patch %s, avg len: %s, act len: %s' % (self.patch_id, len(seg_avg), len(seg_act)))
        
        static_info = ''
        for self.patch_id, static_patch in self.static_patches.items():
            coords_len = len(static_patch['segs'][0]['avg_y_coords'])
            static_info += '%s, ' % coords_len
            if coords_len != len(seg_avg):
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> static and seg len do not match')
        #print('static info: %s' % static_info)
        
        dy_coords = [ ]
        common_pow_2 = self.fft_len
        
        for self.patch_id, static_patch in self.static_patches.items():
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
        self.dy_coord_avgs.append(dy_coord_avg)
        
        dy_fft_of_avg = _calc_fft(dy_coord_avg)
        self.patch_ffts.append(dy_fft_of_avg)
        
        #self._plot_step()
    
    def _plot_fft_avg_of_slice(self):
        avgs = len(self.dy_coord_avgs)
        # 1 second worth at 6 decimation and 30 fps
        step_size = 5
        step_size = 16
        for i in range(step_size, avgs):
            spectra = self._calc_spectra(i - step_size, i)
            fig, axs = plt.subplots(nrows=2, ncols=1)
            for j in range(i - step_size, i):
                fft = self.patch_ffts[j]
                axs[0].plot(np.arange(len(fft)), fft)
            axs[1].plot(np.arange(len(spectra)), spectra)
            plt.show()
    
    def _calc_spectra(self, start_step, end_step):
        spectra = 0
        spectra_count = 0.0
        #for i in range(self.fft_len // 2, len(self.dy_coord_avgs), self.fft_len // 2):
        #    if i + self.fft_len // 2 > len(self.dy_coord_avgs):
        #        break
        print('start_step:', start_step, 'end_step:', end_step)
        for i in range(start_step, end_step):
            print('>>>> len(self.dy_coord_avgs[i]):', len(self.dy_coord_avgs[i]), 'i:', i)
            curr_a = np.multiply(self.dy_coord_avgs[i], self.fft_win)
            curr_b = _calc_fft(curr_a) / self.fft_len * 2.
            curr_c = (curr_b * 2.) ** 2.
            spectra = curr_c + spectra
            spectra_count += 1.0
        spectra = np.sqrt(spectra / spectra_count)
        return spectra
    
    def _plot_fft_avg_of_all(self):
        spectra = 0
        spectra_count = 0.0
        #for i in range(self.fft_len // 2, len(self.dy_coord_avgs), self.fft_len // 2):
        #    if i + self.fft_len // 2 > len(self.dy_coord_avgs):
        #        break
        for i in range(len(self.dy_coord_avgs)):
            print('>>>> len(self.dy_coord_avgs[i]):', len(self.dy_coord_avgs[i]), 'i:', i)
            curr_a = np.multiply(self.dy_coord_avgs[i], self.fft_win)
            curr_b = _calc_fft(curr_a) / self.fft_len * 2.
            curr_c = (curr_b * 2.) ** 2.
            spectra = curr_c + spectra
            spectra_count += 1.0
            if spectra_count >= 5:
                break
        spectra = np.sqrt(spectra / spectra_count)
        fig, axs = plt.subplots(nrows=1, ncols=1)
        axs.plot(np.arange(len(spectra)), spectra)
        plt.show()
        print('self.dy_coord_avgs:', len(self.dy_coord_avgs), 'spectra_count:', spectra_count)
    
    def _plot_step(self):
        if len(self.dy_coord_avgs) >= self.min_seg_len:
            #print('self.patch_ffts: ', len(self.patch_ffts), ', self.min_seg_len:', self.min_seg_len)
            #print('self.patch_ffts[-self.fft_len // 2]:', len(self.patch_ffts[-self.fft_len // 2]), 'len(self.fft_win):', len(self.fft_win))
            #prev_whole = np.multiply(self.dy_coord_avgs[-self.fft_len], self.fft_win)
            prev_half = np.multiply(self.dy_coord_avgs[-self.fft_len // 2], self.fft_win)
            #prev_quarter = np.multiply(self.dy_coord_avgs[-self.fft_len // 4], self.fft_win)
            curr = np.multiply(self.dy_coord_avgs[-1], self.fft_win)
            
            #prev_whole = self.dy_coord_avgs[-self.fft_len]
            #prev_half = self.dy_coord_avgs[-self.fft_len // 2]
            #curr = self.dy_coord_avgs[-1]
            
            fft_0 = _calc_fft(curr)
            fft_1 = _calc_fft((curr + prev_half) / 2.0)
            #fft_2 = _calc_fft((curr + prev_quarter + prev_half) / 3.0)
            #fft_2 = _calc_fft((curr + prev_half + prev_whole) / 3.0)
            
            #fft_3 = _calc_fft((curr * prev_quarter * prev_half * prev_whole) / 4.0)
            
            #prev_whole_dy = self.dy_coord_avgs[-self.fft_len]
            prev_half_dy = self.dy_coord_avgs[-self.fft_len // 2]
            #prev_quarter_dy = self.dy_coord_avgs[-self.fft_len // 4]
            curr_dy = self.dy_coord_avgs[-1]
            
            w0_f, w0_pxx = sp.signal.welch(self.dy_coord_avgs[-1]) #, average='median', nperseg=1024)
            
            '''
            Could try (see https://gist.github.com/kinnala/7775902)
            spectra[itr] = (ampl_corr*np.abs(np.fft.fft(np.hanning(n)*pt[itr,begin:end])[0:n/2])/n*2.)**2+spectra[itr]
            
            - This is example is a bit different in that it shows a way to combine these overlapping things
            
            a = window * data
            b = _calc_fft(a) / self.fft_len * 2.
            c = (b * 2) ** 2
            d = c + spectra
            
            d is spectra ...
            
            ^^^ Do that for how ever many overlaps in for loop
            
            Then do
            
            sqrt(spectra / (2. ** M + 1))
            
            50% overlap of 2^M+1 windows
            '''
            # self.fft_win
            prev_half_a = np.multiply(self.dy_coord_avgs[-self.fft_len // 2], self.fft_win)
            prev_half_b = _calc_fft(prev_half_a) / self.fft_len * 2.
            prev_half_c = (prev_half_b * 2.) ** 2.
            #
            curr_a = np.multiply(self.dy_coord_avgs[-1], self.fft_win)
            curr_b = _calc_fft(curr_a) / self.fft_len * 2.
            curr_c = (curr_b * 2.) ** 2.
            #
            spectra = np.sqrt((prev_half_c + curr_c) / 2.)
            
            fig, axs = plt.subplots(nrows=3, ncols=1)
            axs[0].plot(np.arange(len(curr_dy)), curr_dy)
            #axs[0].plot(np.arange(len(prev_whole_dy)), prev_whole_dy)
            axs[0].plot(np.arange(len(prev_half_dy)), prev_half_dy)
            #axs[0].plot(np.arange(len(prev_quarter_dy)), prev_quarter_dy)
            
            #axs[1].plot(np.arange(len(dy_fft_of_avg)), dy_fft_of_avg)
            #axs[1].plot(np.arange(len(fft_0)), fft_0)
            #axs[1].plot(np.arange(len(fft_1)), fft_1)
            #axs[1].plot(np.arange(len(fft_2)), fft_2)
            
            #axs[2].plot(np.arange(len(w0_f)), w0_f)
            #axs[2].plot(np.arange(len(w0_pxx)), w0_pxx)
            
            #axs[1].plot(np.arange(len(fft_3)), fft_3)
            
            #axs[1].plot(np.arange(len(self.patch_ffts[-self.fft_len])), self.patch_ffts[-self.fft_len])
            axs[1].plot(np.arange(len(self.patch_ffts[-self.fft_len // 2])), self.patch_ffts[-self.fft_len // 2])
            axs[1].plot(np.arange(len(self.patch_ffts[-1])), self.patch_ffts[-1])
            
            #axs[2].plot(np.arange(self.fft_len // 2), prev_whole)
            #axs[2].plot(np.arange(self.fft_len // 2), prev_half)
            #axs[2].plot(np.arange(self.fft_len // 2), prev_quarter)
            #axs[2].plot(np.arange(self.fft_len // 2), curr)
            
            #axs[3].plot(np.arange(self.fft_len // 2), (prev_half * curr) / 2.0)
            #axs[3].plot(np.arange(self.fft_len // 2), (prev_half * prev_quarter * curr) / 3.0)
            #axs[3].plot(np.arange(self.fft_len // 2), (prev_whole * prev_half * curr) / 3.0)
            
            axs[2].plot(np.arange(len(spectra)), spectra)
            
            plt.show()

def _convert_to_np_array(vals: list):
    return np.asarray(vals, dtype=float)

def _calc_fft(y: np.ndarray) -> np.ndarray:
    yf = sp.fft.fft(y)
    yf = 2.0 / len(y) * np.abs(yf[:len(y) // 2])
    return yf





