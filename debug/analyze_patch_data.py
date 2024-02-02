from pet_breathy.debug_things import DebugPatchAnalyzer
from pet_breathy.fixed_data_structures import FixedAvgQueue
from pet_breathy.video_file_reader import VideoFileReader
from pet_breathy.video_display import VideoDisplay
from pet_breathy.segment import Segment
import sys
import os
import json
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import cv2 as cv

'''
Usage:

[josh@josh-pc pet-breathy]$ pwd
/home/josh/Documents/projects/breath_counter/pet-breathy
PYTHONPATH=$(pwd) python3

analyzer = DebugPatchAnalyzer('./debug/data/Blaze2_patch_data.json')
analyzer = DebugPatchAnalyzer('./debug/data/Blaze2_line_patch_data.json')
analyzer = DebugPatchAnalyzer('./debug/data/not_sure_which_vid_this_was_from_patch_data.json')
analyzer = DebugPatchAnalyzer('./debug/data/20240217_212535_patch_data.json')

exec(open('./debug/analyze_patch_data.py').read())
'''

def main():
    args = sys.argv[1:]
    _run(*args)

def _run(in_data_path):
    analyzer = DebugPatchAnalyzer(in_data_path)

def _output_info(analyzer):
    print('Number of patch steps: %s' % len(analyzer.patch_steps))
    
    for patch_step in analyzer.patch_steps:
        static_patches = patch_step['static_patches']
        movable_patches = patch_step['movable_patches']
        
        print('static patches len:  %s' % len(static_patches))
        for group_id, static_patch in static_patches.items():
            for seg in static_patch['segs']:
                print('- seg len: %s' % len(seg))
        print('movable patches len: %s' % len(movable_patches))
        for group_id, movable_patch in movable_patches.items():
            print('- group id: %s, seg len: %s' % (group_id, len(movable_patch['segs'][0])))
        break

def _look_for_steps_to_analyze(analyzer, min_fft_size=16):
    good_steps = [ ]
    for patch_idx, patch_step in enumerate(analyzer.patch_steps):
        static_patches = patch_step['static_patches']
        movable_patches = patch_step['movable_patches']
        
        static_patches_good = False
        movable_patches_good = False
        
        for group_id, static_patch in static_patches.items():
            for seg in static_patch['segs']:
                if len(seg['avg_y_coords']) >= min_fft_size:
                    static_patches_good = True
                    break
            if static_patches_good:
                break
        
        for group_id, movable_patch in movable_patches.items():
            for seg in movable_patch['segs']:
                if len(seg['avg_y_coords']) >= min_fft_size:
                    movable_patches_good = True
                    break
            if movable_patches_good:
                break
        
        if static_patches_good and movable_patches_good:
            good_steps.append(patch_idx)
    
    print('Number of steps:      %s' % len(analyzer.patch_steps))
    print('Number of good steps: %s' % len(good_steps))
    print(good_steps)

def _get_patches(patch_dict, min_fft_size):
    patches = { }
    for group_id, patch in patch_dict.items():
        for seg in patch['segs']:
            if len(seg['avg_y_coords']) >= min_fft_size:
                patches[group_id] = patch
    return patches

def _get_latest_buf(buf: np.ndarray, count: int) -> np.ndarray:
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
        avg_y_coords = _get_latest_buf(self.avg_y_coords, count)
        act_y_coords = _get_latest_buf(self.act_y_coords, count)
        return avg_y_coords, act_y_coords
    
    def get_size(self):
        return len(self.avg_y_coords)

def _calc_fft(y: np.ndarray) -> np.ndarray:
    yf = sp.fft.fft(y)
    yf = 2.0 / len(y) * np.abs(yf[:len(y) // 2])
    return yf

def _examine_patch_step(analyzer, step_idx, min_fft_size):
    step = analyzer.patch_steps[step_idx]
    
    static_patches = _get_patches(step['static_patches'], min_fft_size)
    movable_patches = _get_patches(step['movable_patches'], min_fft_size)
    
    print('Num static patches:  %s' % len(static_patches))
    print('Num movable patches: %s' % len(movable_patches))
    
    x = np.arange(min_fft_size)
    
    for movable_patch_id, movable_patch in movable_patches.items():
        movable_segs = [SimpleSeg(s) for s in movable_patch['segs']]
        
        fig, axs = plt.subplots(nrows=3, ncols=3)
        axs[0][1].set_ylim([0, 2])
        axs[1][1].set_ylim([0, 2])
        
        for movable_seg in movable_segs:
            movable_avg_y_coords, movable_act_y_coords = movable_seg.get_latest(min_fft_size)
            
            #print('Static patch count: %s' % len(static_patches))
            
            for static_patch_id, static_patch in static_patches.items():
                static_seg = SimpleSeg(static_patch['segs'][0])
                static_avg_y_coords, static_act_y_coords = static_seg.get_latest(min_fft_size)
                
                avg_diff_y_coord = static_avg_y_coords - movable_avg_y_coords
                
                #avg_kernel_size = 15
                #avg_diff_y_coord_cooled = FixedAvgQueue(len(avg_diff_y_coord), float, avg_kernel_size)
                #avg_diff_y_coord_cooled2 = FixedAvgQueue(len(avg_diff_y_coord), float, avg_kernel_size)
                #for v in avg_diff_y_coord:
                #    avg_diff_y_coord_cooled.append(v)
                #for v in avg_diff_y_coord_cooled.get_avg_buf():
                #    avg_diff_y_coord_cooled2.append(v)
                #for v in avg_diff_y_coord_cooled2.get_avg_buf():
                #    avg_diff_y_coord_cooled.append(v)
                
                #avg_diff_y_coord *= 1
                dy_coord = (static_act_y_coords - movable_act_y_coords) - avg_diff_y_coord
                #dy_coord = (static_act_y_coords - movable_act_y_coords) - avg_diff_y_coord_cooled.get_avg_buf() #avg_diff_y_coord
                fft_coord = _calc_fft(dy_coord)
                
                
                
                #fig, axs = plt.subplots(nrows=3, ncols=3)
                #axs[0][1].set_ylim([0, 2])
                #axs[1][1].set_ylim([0, 2])
                ##axs[2][1].set_ylim([0, 2])
                
                alpha = .35
                
                axs[0][0].plot(x, dy_coord, alpha=alpha)
                axs[0][1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha)
                axs[1][0].plot(x, static_act_y_coords - movable_act_y_coords, alpha=alpha)
                axs[1][0].plot(x, static_avg_y_coords - movable_avg_y_coords, alpha=alpha)
                axs[1][0].plot(x, avg_diff_y_coord_cooled.get_avg_buf(), alpha=alpha)
                
                fft_coord = _calc_fft(static_act_y_coords - movable_act_y_coords)
                axs[1][1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha)
                
                fft_coord = _calc_fft(static_avg_y_coords - movable_avg_y_coords)
                axs[1][1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha)
                
                dy_coord = (static_act_y_coords - movable_act_y_coords) # * avg_diff_y_coord
                dy_coord = dy_coord - dy_coord[0]
                fft_coord = _calc_fft(dy_coord)
                
                axs[2][0].plot(x, dy_coord, alpha=alpha)
                axs[2][1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha)
                
                #for static_patch_id, static_patch in static_patches.items():
                for static_patch_id_j, static_patch_j in static_patches.items():
                    static_seg_j = SimpleSeg(static_patch_j['segs'][0])
                    static_avg_y_coords_j, static_act_y_coords_j = static_seg_j.get_latest(min_fft_size)
                    
                    axs[2][2].plot(x, static_avg_y_coords_j[0] - static_avg_y_coords_j, alpha=.30)
                    axs[2][2].plot(x, static_act_y_coords_j[0] - static_act_y_coords_j, alpha=.30)
                    
                    if static_patch_id_j == static_patch_id:
                        continue
                    
                    axs[0][2].plot(
                        x,
                        (static_avg_y_coords[0] - static_avg_y_coords) - (static_avg_y_coords_j[0] - static_avg_y_coords_j),
                        alpha=alpha)
                    axs[1][2].plot(
                        x,
                        (static_act_y_coords[0] - static_act_y_coords) - (static_act_y_coords_j[0] - static_act_y_coords_j),
                        alpha=alpha)
                
        plt.show()

if False:
    _examine_patch_step(analyzer, 100, 64)
    _examine_patch_step(analyzer, 0, 16)

'''
analyzer = DebugPatchAnalyzer('./debug/data/20240217_212535_patch_data.json')
analyzer = DebugPatchAnalyzer('./debug/data/Blaze2_patch_data.json')
analyzer = DebugPatchAnalyzer('./debug/data/not_sure_which_vid_this_was_from_patch_data.json')
'''

def _display_point_info(video_path, frame_num, points, patch_ids):
    reader = VideoFileReader(video_path)
    display = VideoDisplay('Debug', reader.info.width, reader.info.height, 0.5)
    for i in range(frame_num + 1):
        frame = reader.get_next_frame()
        
        if i == frame_num:
            break
    
    for idx, (x,y) in enumerate(points):
        cv.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1)
        
        cv.putText(frame,               # img
            '%s' % patch_ids[idx],      # text
            (x, y),                     # org
            cv.FONT_HERSHEY_SIMPLEX,    # fontFace
            .5,                         # fontScale
            (255,255,255),              # color
            1,                          # thickness
            cv.LINE_AA)                 # lineType
    
    display.show(frame)
    
    while display.is_open():
        key = cv.waitKey(50)
        if key == 99 or key == 100:
            break
    
    display.close()
    
    return key

if False:
    video_file_name = os.path.basename(analyzer.json_path).rstrip('_patch_data.json') + '.mp4'
    video_dir = '../videos'
    video_path = os.path.join(video_dir, video_file_name)

if False:    
    _display_point_info(video_path, 0, [[50, 50]])

def _find_patches(analyzer, step_idx_to_examine=-1):
    for step_idx, step in enumerate(analyzer.patch_steps):
        if step_idx_to_examine != -1 and step_idx != step_idx_to_examine:
            continue
        static_patches = _get_patches(step['static_patches'], 64)
        movable_patches = _get_patches(step['movable_patches'], 64)
        frame_num = int(step['frame_num'])
        
        if len(static_patches) > 0 and len(movable_patches) > 0:
            print('step: %s, frame #: %s, # static: %s, # movable: %s' % (
                step_idx,
                frame_num,
                len(static_patches),
                len(movable_patches)))
            
            # Can uncomment this to find patches of interest
            #continue
            
            for movable_patch_id, movable_patch in movable_patches.items():
                x, y = movable_patch['points'][0]
                
                print('Movable patch id: %s' % movable_patch_id)
                key = _display_point_info(video_path, frame_num, [[int(x), int(y)]])
                if key == 100:
                    return
            break

if False:
    _find_patches(analyzer, 71)

def _display_patches(analyzer, step_idx, fft_size, in_patch_ids):
    print('Num of steps: %s' % len(analyzer.patch_steps))
    step = analyzer.patch_steps[step_idx]
    static_patches = _get_patches(step['static_patches'], fft_size)
    movable_patches = _get_patches(step['movable_patches'], fft_size)
    
    frame_num = int(step['frame_num'])
    print('Frame num: %s' % frame_num)
    
    points = [ ]
    out_patch_ids = [ ]
    for movable_patch_id, movable_patch in movable_patches.items():
        if len(in_patch_ids) == 0 or int(movable_patch_id) in in_patch_ids:
            x, y = movable_patch['points'][0]
            print('{ %s, %s }' % (x, y))
            points.append([int(x), int(y)])
            out_patch_ids.append(movable_patch_id)
    
    _display_point_info(video_path, frame_num, points, out_patch_ids)

if False:
    _display_patches(analyzer, 71, 64, [22, 48, 95, 115, 27, 32, 42, 63, 119, 87, 98, 108, 124, 133, 130])
    
    # Looking for other points ... not finding any where /home/josh/Documents/projects/breath_counter/pet-breathy/debug/videos/20240217_212535_output.mp4 had some good points
    # The video is different than this data set ... or the data set and the video do not go together
    _display_patches(analyzer, 100, 64, [ ])

def _examine_patch(analyzer, step_idx, frame_num, patch_id, fft_size):
    step = analyzer.patch_steps[step_idx]
    static_patches = _get_patches(step['static_patches'], fft_size)
    movable_patches = _get_patches(step['movable_patches'], fft_size)
    
    print('step: %s, frame #: %s, # static: %s, # movable: %s' % (
        step_idx,
        frame_num,
        len(static_patches),
        len(movable_patches)))
    
    movable_patch = movable_patches[str(patch_id)]
    #segs = movable_patch['segs']
    #points = movable_patch['points']
    
    movable_segs = [SimpleSeg(s) for s in movable_patch['segs']]
    
    x,y = movable_patch['points'][0]
    key = _display_point_info(video_path, frame_num, [[int(x), int(y)]])
    #if key == 100:
    #    return
    
    fig, axs = plt.subplots(nrows=3, ncols=3)
    axs[0][1].set_ylim([0, 2])
    axs[1][1].set_ylim([0, 2])
    
    x = np.arange(fft_size)
    
    for movable_seg in movable_segs:
        movable_avg_y_coords, movable_act_y_coords = movable_seg.get_latest(fft_size)
        
        #print('Static patch count: %s' % len(static_patches))
        
        for static_patch_id, static_patch in static_patches.items():
            static_seg = SimpleSeg(static_patch['segs'][0])
            static_avg_y_coords, static_act_y_coords = static_seg.get_latest(fft_size)
            
            avg_diff_y_coord = static_avg_y_coords - movable_avg_y_coords
            dy_coord = (static_act_y_coords - movable_act_y_coords) - avg_diff_y_coord
            fft_coord = _calc_fft(dy_coord)
                
            alpha = .35
            
            axs[0][0].plot(x, dy_coord, alpha=alpha)
            axs[0][1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha)
            axs[1][0].plot(x, static_act_y_coords - movable_act_y_coords, alpha=alpha)
            axs[1][0].plot(x, static_avg_y_coords - movable_avg_y_coords, alpha=alpha)
            
            fft_coord = _calc_fft(static_act_y_coords - movable_act_y_coords)
            axs[1][1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha)
            
            fft_coord = _calc_fft(static_avg_y_coords - movable_avg_y_coords)
            axs[1][1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha)
            
            dy_coord = (static_act_y_coords - movable_act_y_coords) # * avg_diff_y_coord
            dy_coord = dy_coord - dy_coord[0]
            fft_coord = _calc_fft(dy_coord)
            
            axs[2][0].plot(x, dy_coord, alpha=alpha)
            axs[2][1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha)
            
            #for static_patch_id, static_patch in static_patches.items():
            for static_patch_id_j, static_patch_j in static_patches.items():
                static_seg_j = SimpleSeg(static_patch_j['segs'][0])
                static_avg_y_coords_j, static_act_y_coords_j = static_seg_j.get_latest(fft_size)
                
                axs[2][2].plot(x, static_avg_y_coords_j[0] - static_avg_y_coords_j, alpha=.30)
                axs[2][2].plot(x, static_act_y_coords_j[0] - static_act_y_coords_j, alpha=.30)
                
                if static_patch_id_j == static_patch_id:
                    continue
                
                axs[0][2].plot(
                    x,
                    (static_avg_y_coords[0] - static_avg_y_coords) - (static_avg_y_coords_j[0] - static_avg_y_coords_j),
                    alpha=alpha)
                axs[1][2].plot(
                    x,
                    (static_act_y_coords[0] - static_act_y_coords) - (static_act_y_coords_j[0] - static_act_y_coords_j),
                    alpha=alpha)
    plt.show()

# Looking to see if there's a better estimate the camera movement
# Trying to prevent false positives, is it due to the camera? Due to me breathing which effected the phone movement to have a signal on various points
# ../videos/20240217_212535.mp4
def _examine_patch_v2(analyzer, step_idx, frame_num, patch_id, fft_size):
    step = analyzer.patch_steps[step_idx]
    static_patches = _get_patches(step['static_patches'], fft_size)
    movable_patches = _get_patches(step['movable_patches'], fft_size)
    
    print('step: %s, frame #: %s, # static: %s, # movable: %s' % (
        step_idx,
        frame_num,
        len(static_patches),
        len(movable_patches)))
    
    fig0, axs0 = plt.subplots(nrows=5, ncols=2)
    fig1, axs1 = plt.subplots(nrows=1, ncols=2)
    
    x = np.arange(fft_size)
    
    movable_patch = movable_patches[str(patch_id)]
    movable_segs = [SimpleSeg(s) for s in movable_patch['segs']]
    
    for static_patch_id, static_patch in static_patches.items():
        static_patch_id = int(static_patch_id)
        static_seg = SimpleSeg(static_patch['segs'][0])
        static_avg_y_coords, static_act_y_coords = static_seg.get_latest(fft_size)
        
        static_points = static_patch['points'][0]
        
        alpha = .35
        
        axs0[0][0].plot(
            x,
            static_act_y_coords[0] - static_act_y_coords,
            alpha=alpha)
        axs0[0][1].plot(
            x,
            static_avg_y_coords[0] - static_avg_y_coords,
            alpha=alpha,
            label='%s %s' % (int(static_points[0]), int(static_points[1])))
        
        for movable_seg_idx, movable_seg in enumerate(movable_segs):
            movable_avg_y_coords, movable_act_y_coords = movable_seg.get_latest(fft_size)
            
            axs0[static_patch_id + 1][0].plot(
                x, movable_act_y_coords[0] - movable_act_y_coords, alpha=alpha)
            axs0[static_patch_id + 1][1].plot(
                x, movable_avg_y_coords[0] - movable_avg_y_coords, alpha=alpha)
            
            axs0[static_patch_id + 1][0].plot(
                x,
                static_act_y_coords[0] - static_act_y_coords,
                alpha=alpha)
            axs0[static_patch_id + 1][1].plot(
                x,
                static_avg_y_coords[0] - static_avg_y_coords,
                alpha=alpha)
                #label='%s %s' % (int(static_points[0]), int(static_points[1])))
            
            if movable_seg_idx == 0 and \
                    (static_patch_id == 0 or static_patch_id == 2):
                    #(static_patch_id == 1 or static_patch_id == 3):
                    #static_patch_id == 1:
                    #static_patch_id >= 0: #(static_patch_id == 0 or static_patch_id == 2):
                
                #avg_diff_y_coord = static_avg_y_coords - movable_avg_y_coords
                #dy_coord = (static_act_y_coords - movable_act_y_coords) - avg_diff_y_coord
                dy_coord = movable_act_y_coords - movable_avg_y_coords
                
                fft_coord = _calc_fft(dy_coord)
                
                dy_seg = Segment(1024, 7)
                for y in dy_coord:
                    dy_seg.append_y(y)
                
                dy_seg_avg = dy_seg.get_latest_avg_y_coords(fft_size)
                dy_seg_fft_avg = _calc_fft(dy_seg_avg)
                
                axs1[0].plot(x, dy_coord, alpha=alpha)
                axs1[0].plot(x, dy_seg_avg, alpha=alpha, label='avg')
                axs1[1].plot(np.arange(len(fft_coord)), fft_coord, alpha=alpha, label=str(static_patch_id))
                axs1[1].plot(np.arange(len(dy_seg_fft_avg)), dy_seg_fft_avg, alpha=alpha, label='avg y fft')
    
    #axs0[0].legend(loc="upper right")
    axs0[0][1].legend(loc="upper right")
    axs1[0].legend(loc="upper right")
    axs1[1].legend(loc="upper right")
    plt.show()

def _examine_patch_v3(analyzer, step_idx, frame_num, patch_id, fft_size):
    step = analyzer.patch_steps[step_idx]
    static_patches = _get_patches(step['static_patches'], fft_size)
    movable_patches = _get_patches(step['movable_patches'], fft_size)
    
    print('step: %s, frame #: %s, # static: %s, # movable: %s' % (
        step_idx,
        frame_num,
        len(static_patches),
        len(movable_patches)))
    
    #fig0, axs0 = plt.subplots(nrows=1, ncols=2)
    #fig1, axs1 = plt.subplots(nrows=4, ncols=2)
    fig2, axs2 = plt.subplots(nrows=1, ncols=2)
    
    x = np.arange(fft_size)
    alpha = .35
    
    static0_patch = static_patches['0']
    static0_seg = SimpleSeg(static0_patch['segs'][0])
    static0_avg_y_coords, static0_act_y_coords = static0_seg.get_latest(fft_size)
    static0_points = static0_patch['points'][0]
    static0_diff = static0_act_y_coords - static0_avg_y_coords
    static0_fft = _calc_fft(static0_diff)
    
    static1_patch = static_patches['1']
    static1_seg = SimpleSeg(static1_patch['segs'][0])
    static1_avg_y_coords, static1_act_y_coords = static1_seg.get_latest(fft_size)
    static1_points = static1_patch['points'][0]
    static1_diff = static1_act_y_coords - static1_avg_y_coords
    static1_fft = _calc_fft(static1_diff)
    
    static2_patch = static_patches['2']
    static2_seg = SimpleSeg(static2_patch['segs'][0])
    static2_avg_y_coords, static2_act_y_coords = static2_seg.get_latest(fft_size)
    static2_points = static2_patch['points'][0]
    static2_diff = static2_act_y_coords - static2_avg_y_coords
    static2_fft = _calc_fft(static2_diff)
    
    static3_patch = static_patches['3']
    static3_seg = SimpleSeg(static3_patch['segs'][0])
    static3_avg_y_coords, static3_act_y_coords = static3_seg.get_latest(fft_size)
    static3_points = static3_patch['points'][0]
    static3_diff = static3_act_y_coords - static3_avg_y_coords
    static3_fft = _calc_fft(static3_diff)
    
    if False:
        axs1[0][0].plot(x, static0_diff)
        axs1[0][1].plot(np.arange(len(static0_fft)), static0_fft)
        axs1[1][0].plot(x, static1_diff)
        axs1[1][1].plot(np.arange(len(static1_fft)), static1_fft)
        axs1[2][0].plot(x, static2_diff)
        axs1[2][1].plot(np.arange(len(static2_fft)), static2_fft)
        axs1[3][0].plot(x, static3_diff)
        axs1[3][1].plot(np.arange(len(static3_fft)), static3_fft)
    
    #static_sum = (static0_diff + static1_diff + static2_diff + static3_diff) # / 4.0
    #static_fft = _calc_fft(static_sum)
    #axs0[0].plot(x, static_sum)
    #axs0[1].plot(np.arange(len(static_fft)), static_fft)
    
    #static_avg_diff = static0_avg_y_coords - static2_avg_y_coords
    #static_coord = (static0_act_y_coords - static2_act_y_coords) - static_avg_diff
    #static_fft = _calc_fft(static_coord)
    #axs0[0].plot(x, static_coord)
    #axs0[1].plot(np.arange(len(static_fft)), static_fft)
    
    movable_patch = movable_patches[str(patch_id)]
    movable_segs = [SimpleSeg(s) for s in movable_patch['segs']]
    
    for movable_seg_idx, movable_seg in enumerate(movable_segs):
        movable_avg_y_coords, movable_act_y_coords = movable_seg.get_latest(fft_size)
        
        #movable_diff = movable_act_y_coords - movable_avg_y_coords
        #movable_fft = _calc_fft(movable_diff)
        
        #movable_avg_diff = ((static0_avg_y_coords + static2_avg_y_coords) / 2.0) - movable_avg_y_coords
        #movable_coord = (((static0_act_y_coords + static2_act_y_coords) / 2.0) - movable_act_y_coords) - movable_avg_diff
        #movable_fft = _calc_fft(movable_coord)
        
        '''
        - There appears to be 2 signals
        - If I look at a point on blaze, I see 2 main signals
            - Signal 1 strength 1.75
            - Signal 2 strength 1.25
        - If I look at a point not on blaze, I see one weak signal, .8 signal strength
        '''
        
        avg_diff_y_coord = static0_avg_y_coords - movable_avg_y_coords
        dy_coord = (static0_act_y_coords - movable_act_y_coords) - avg_diff_y_coord
        dy_fft = _calc_fft(dy_coord)
        axs2[0].plot(x, dy_coord)
        axs2[1].plot(np.arange(len(dy_fft)), dy_fft)
        
        #movable_diff -= static_sum
        #movable_fft = _calc_fft(movable_diff) - static_fft
        
        #axs2[0].plot(x, movable_diff)
        #axs2[1].plot(np.arange(len(movable_fft)), movable_fft)
        
        #axs2[0].plot(x, movable_coord)
        #axs2[1].plot(np.arange(len(movable_fft)), movable_fft)
        
        #movable_avg_diff = ((static0_avg_y_coords) / 1.0) - movable_avg_y_coords
        #movable_coord = (((static0_act_y_coords) / 1.0) - movable_act_y_coords) - movable_avg_diff
        #movable_fft = _calc_fft(movable_coord)
        
        #axs2[0].plot(x, movable_coord)
        #axs2[1].plot(np.arange(len(movable_fft)), movable_fft)
        
        #val = movable_diff - static0_diff
        #fft_val = _calc_fft(val)
        #axs2[0].plot(x, val)
        #axs2[1].plot(np.arange(len(fft_val)), fft_val)
        
        #val = movable_diff - static2_diff
        #fft_val = _calc_fft(val)
        #axs2[0].plot(x, val)
        #axs2[1].plot(np.arange(len(fft_val)), fft_val)
        
        #val = static0_avg_y_coords - np.mean(static0_avg_y_coords)
        #fft_val = _calc_fft(val)
        #axs2[0].plot(x, val)
        #axs2[1].plot(np.arange(len(fft_val)), fft_val)
        break
    
    plt.show()

def _foo1():
    # 20240217_212535_patch_data.json
    step_idx = 71
    frame_num = 510
    patch_id = 131 #22 #27
    fft_size = 64
    #_examine_patch(analyzer, step_idx, frame_num, patch_id, fft_size)
    _examine_patch_v2(analyzer, step_idx, frame_num, patch_id, fft_size)

def _foo2():
    # Blaze2_patch_data.json
    step_idx = 72
    frame_num = 516
    patch_id = 81 #128
    fft_size = 64
    #_examine_patch(analyzer, step_idx, frame_num, patch_id, fft_size)
    #_examine_patch_v2(analyzer, step_idx, frame_num, patch_id, fft_size)
    _examine_patch_v3(analyzer, step_idx, frame_num, patch_id, fft_size)

if False:
    exec(open('./debug/analyze_patch_data.py').read())

'''
Blaze2_line_patch_data.json


Blaze2_patch_data.json (step 72, frame# 516)
- Decent patch ids
    - 128, 161          # closest to top left static point
- Points not on Blaze
    - 81, 112, 129      # closest to top right static point
    - 97                # closest to bottom left static point

Point 81 has highest correlation with top right and top right static points, but better correlation with top right static point


20240217_212535_patch_data.json
- Step: 71, frame # 510
    - Wagner patch ids
        - 22, 48, 95, 115, 27, 32, 42, 63, 119, 87, 98, 108, 124, 133, 130      <- All
        - 22, 48, 95, 115           <- middle right
        - 27                        <- middle top, this looks like a good point
        - 32, 42, 63, 119           <- middle bottom, bad location
        - 87, 98, 108, 124, 133     <- left bottom, bad location
        - 130                       <- top right
    - Not Wagner patch ids
        - 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 28, 29, 30, 33, 34, 36, 37, 38, 39, 40, 41, 43, 45, 46, 47, 49, 51, 52, 54, 56, 57, 58, 59, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 85, 88, 89, 90, 91, 92, 93, 94, 96, 97, 99, 100, 101, 102, 104, 106, 107, 109, 110, 111, 112, 113, 117, 118, 120, 121, 123, 125, 126, 127, 128, 129, 131, 132, 134, 135

- Step: 48, frame # 372
    - Wagner patch ids
        - 98                <- This is on his bottom side, doesn't look like a good location
                               This point looks like garbage when I plot it
    - Not Wagner patch ids
        - 11
        - 13
        - 16
        - 49
'''

#if __name__ == '__main__':
#    main()
