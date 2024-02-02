from pet_breathy.video_display import VideoDisplay
from pet_breathy.video_info import VideoInfo
from pet_breathy.video_file_reader import VideoFileReader
from pet_breathy.point import Point
from pet_breathy.segment import Segment
from pet_breathy.signal_analyzer import SignalAnalyzer

import numpy as np
import cv2 as cv
import json
from matplotlib import pyplot as plt
import os

'''
- Display location of static points
- Display location of patches / points
'''

class DebugThing:
    def __init__(self, info: VideoInfo, frame: np.ndarray):
        self.info = info
        self.display = VideoDisplay('Debug', info.width, info.height, 0.5)
        self.display.move(20, 20)
        
        self.top_right_text_pos = Point(
            info.width - 550,
            50)
        
        self.background_for_bpm_pos = Point(
            self.top_right_text_pos.x - 10,
            self.top_right_text_pos.y - 35)
        self.background_for_bpm_width = 525
        self.background_for_bpm_height = 50
        
        self.draw_mask = np.zeros_like(frame)
        
        self.frame = None
        self.sorted_patches_with_signal = None
        
        self.enable_output_video = False
        self.output_path = './debug/videos/debug.mp4' #.avi'
        if self.enable_output_video:
            self.output_writer = cv.VideoWriter(
                self.output_path,
                # Use MJPG for .avi, MPV4 didn't work for .mp4, rather, XVID
                # did work to output gray scale .mp4
                cv.VideoWriter_fourcc(*'XVID'), #(*'MJPG'), #(*'MPV4'),
                self.info.fps // 6,
                (self.info.width, self.info.height),
                False)

    def set_frame(self, frame: np.ndarray):
        self.frame = frame.copy()

    def done(self):
        if self.enable_output_video:
            self.output_writer.release()

    def do_stuff(self, frame_num):
        cv.putText(self.frame,
            'Frame %s' % frame_num,
            (0,25),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv.LINE_AA)

        for static_patch in self.static_patches:
            point = static_patch.get_point_group().get_points()[0]
            cv.circle(
                self.frame,
                (int(point[0]), int(point[1])),
                25,
                (255, 255, 255),
                -1)
        
        for idx, patch in enumerate(self.sorted_patches_with_signal):
            point_group = patch.get_point_group()
            points = point_group.get_points()
            
            if len(points) > 0:
                point = points[0]
                x = int(point[0])
                y = int(point[1])
                
                # TODO Can color code the points based on some criteria
                
                cv.circle(
                    self.frame,
                    (x, y),
                    5,
                    (255, 255, 255),
                    -1)
                
                cv.putText(self.frame,
                    str(patch.get_id()),
                    (x, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv.LINE_AA)
                
                if idx == 0:
                    cv.line(
                        self.frame,
                        (x, y),
                        (self.background_for_bpm_pos.x,
                            self.background_for_bpm_pos.y + (self.background_for_bpm_height // 2)),
                        (255, 255, 255),
                        3)
                    
                    signal = patch.get_debug_signal()
                    bpm_msg = ''
                    if signal:
                        bpm_msg = 'bpm: %.2f, sig: %.2f, fft: %s' % (
                            signal.bpm_est, signal.strength, signal.fft_size)
                    
                    cv.rectangle(
                        self.frame,
                        (self.background_for_bpm_pos.x, self.background_for_bpm_pos.y),
                        (self.background_for_bpm_pos.x + self.background_for_bpm_width,
                            self.background_for_bpm_pos.y + self.background_for_bpm_height),
                        (255,255,255),
                        -1)
                    
                    if bpm_msg:
                        cv.putText(
                            self.frame,
                            bpm_msg,
                            (self.top_right_text_pos.x, self.top_right_text_pos.y),
                            cv.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,0),
                            2,
                            cv.LINE_AA)

        self.display.show(self.frame)
        
        if self.enable_output_video:
            self.output_writer.write(self.frame)


# python3 pet_breathy_cli.py --in_video_path ../videos/PXL_20230825_040038487.mp4
# python3 analyze_patch_data.py ./data/patch_data.json
class DebugPatchCollector:
    def __init__(self):
        self.max_seg_len = 0
        self.avg_kernel_len = 0
        self.movable_patch_dist = 0
        self.video_info = VideoInfo()
        self.decimation = 0
        
        self.patch_steps = [ ]
    
    def add_patches(self, frame_num, static_patches, movable_patches):
        if not self.patch_steps:
            # Collect basic info on first patch step
            self.max_seg_len = static_patches[0].seg.y.get_capacity()
            self.avg_kernel_len = static_patches[0].seg.y.kernel_size
            self.movable_patch_dist = movable_patches[0].point_dist
        
        self._add_patch_step(frame_num, static_patches, movable_patches)

    def output_to_file(self, path):
        info = {
            'video_info': {
                'fps': self.video_info.fps,
                'width': self.video_info.width,
                'height': self.video_info.height,
                'frame_count': self.video_info.frame_count,
            },
            'patch_info': {
                'max_seg_len': self.max_seg_len,
                'movable_patch_dist': self.movable_patch_dist,
            },
            'signal_info': {
                'decimation': self.decimation,
                'avg_kernel_len': self.avg_kernel_len,
            },
            'patch_steps': self.patch_steps,
        }
        with open(path, 'w') as fh:
            fh.write(json.dumps(info, indent=4))

    def _add_patch_step(self, frame_num, static_patches, movable_patches):
        # Patch dictionaries
        #   Key:      patch id
        #   Value:    { 'patch_id': <patch id>, 'points': [ ... ] }
        patch_step = {
            'frame_num' : frame_num,
            'static_patches' : { },
            'movable_patches' : { },
        }
        
        self.patch_steps.append(patch_step)
        
        json_static_patches = patch_step['static_patches']
        json_movable_patches = patch_step['movable_patches']
        
        for static_patch in static_patches:
            self._add_static_patch(json_static_patches, static_patch)
        
        for movable_patch in movable_patches:
            self._add_movable_patch(json_movable_patches, movable_patch)

    def _add_static_patch(self, patches, patch):
        self._add_patch(patches, patch)
    
    def _add_movable_patch(self, patches, patch):
        self._add_patch(patches, patch)

    def _add_patch(self, patches, patch):
        patch_id = patch.get_id()
        points = patch.get_point_group().get_points()
        
        patches[patch_id] = {
            'segs' : [ ],
            'points' : [ ]
        }
        
        segs = patches[patch_id]['segs']
        for seg in patch.get_segs():
            self._add_segment(segs, seg)

        points = patches[patch_id]['points']
        for point in patch.get_point_group().get_points():
            points.append([float(point[0]), float(point[1])])

    def _add_segment(self, segs, seg):
        avg_y_coords = seg.get_avg_y_coords()
        act_y_coords = seg.get_act_y_coords()
        
        segs.append({
            'avg_y_coords': list(avg_y_coords),
            'act_y_coords': list(act_y_coords)
        })

class DebugPatchAnalyzer:
    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path) as fh:
            info = json.load(fh)
        
        self.video_info = VideoInfo()
        self.video_info.fps = info['video_info']['fps']
        self.video_info.width = info['video_info']['width']
        self.video_info.height = info['video_info']['height']
        self.video_info.frame_count = info['video_info']['frame_count']
        
        patch_info = info['patch_info']
        self.max_seg_len = int(patch_info['max_seg_len'])
        self.movable_patch_dist = int(patch_info['movable_patch_dist'])
        
        signal_info = info['signal_info']
        self.decimation = int(signal_info['decimation'])
        self.avg_kernel_len = int(signal_info['avg_kernel_len'])
        
        self.patch_steps = info['patch_steps']
        
        '''
        for patch in info['movable_patches'].values():
            num_points_per_movable_patch = len(patch['segs'])
            break
        
        max_points = len(info['static_patches']) + \
            len(info['movable_patches']) * num_points_per_movable_patch
        
        print('max points:               %s' % max_points)
        print('points per movable patch: %s' % num_points_per_movable_patch)
        
        self.fft_size = 16
        
        static_patches = [ ]
        movable_patches = [ ]

        for group_id, patch in info['static_patches'].items():
            static_patch = StaticPatchJson(
                group_id, max_seg_len, avg_kernel_len, self.fft_size, patch['points'])
            static_patches.append(static_patch)
        
        for group_id, patch in info['movable_patches'].items():
            movable_patch = MovablePatchJson(
                group_id, max_seg_len, avg_kernel_len, self.fft_size, patch['segs'])
            movable_patches.append(movable_patch)
        
        self.analyzer = SignalAnalyzer(video_info.fps, decimation)
        
        inspect = True
        for movable_patch in movable_patches:
            if movable_patch.group_id == 9999999:
                inspect = True
                print('group id: %s' % movable_patch.group_id)
                
                self._inspect_movable_patch(movable_patch)
            
            if inspect:
                self._inspect_movable_vs_static_patches(movable_patch, static_patches)
                #break
        plt.show()
        '''
    '''
    def _inspect_movable_patch(self, movable_patch):
        point_seg = movable_patch.point_segs[0]
        
        dx = [ ]
        dy = [ ]
        dists = [ ]
        for i in range(1, len(point_seg)):
            p0 = point_seg[i - 1]
            p1 = point_seg[i]
            
            dist = np.sqrt((p1.y - p0.y) ** 2 + (p1.x - p0.x) ** 2)
            dists.append(dist)
        
        print('min dist: %s' % np.min(dists))
        print('max dist: %s' % np.max(dists))
        print('avg dist: %s' % np.average(dists))
        plt.plot(np.arange(len(dists)), dists)
        plt.show()
    
    def _inspect_movable_vs_static_patches(self, movable_patch, static_patches):
        dy_coords = [ ]
        
        movable_seg = movable_patch.segs[0]
        
        # See def can_we_search(static_points, surrounding_points, poi):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        
        for static_patch in static_patches:
            static_seg = static_patch.seg
            
            static_avg_y_coord = \
                static_seg.get_latest_avg_y_coords(self.fft_size)
            static_act_y_coord = \
                static_seg.get_latest_act_y_coords(self.fft_size)
            
            movable_avg_y_coord = \
                movable_seg.get_latest_avg_y_coords(self.fft_size)
            movable_act_y_coord = \
                movable_seg.get_latest_act_y_coords(self.fft_size)
            
            avg_diff_y_coord = static_avg_y_coord - movable_avg_y_coord
            dy_coord = (static_act_y_coord - movable_act_y_coord) - avg_diff_y_coord
            
            #ax[0][0].plot(np.arange(len(static_avg_y_coord)), static_avg_y_coord - static_avg_y_coord[0])
            #ax[0][1].plot(np.arange(len(static_act_y_coord)), static_act_y_coord - static_act_y_coord[0])
            
            ax[0][0].plot(np.arange(len(movable_avg_y_coord)), movable_avg_y_coord - movable_avg_y_coord[0])
            ax[0][1].plot(np.arange(len(movable_act_y_coord)), movable_act_y_coord - movable_act_y_coord[0])
            
            #ax[0].plot(np.arange(len(movable_avg_y_coord)), movable_avg_y_coord)
            #ax[1].plot(np.arange(len(movable_act_y_coord)), movable_act_y_coord)
            ax[1][0].plot(np.arange(len(dy_coord)), dy_coord)
            
            dy_coords.append(dy_coord)
        
        avg_fft = self.analyzer.calc_avg_fft(dy_coords)
        peaks = self.analyzer.find_peaks(avg_fft)
        sig = self.analyzer.calc_signal(dy_coords, avg_fft, peaks)
        
        ax[1][1].plot(np.arange(len(avg_fft)), avg_fft)
        ax[1][1].set_ylim([0, 8])
        plt.show()
        
        if sig.strength > 100:
            print('movable group id: %s' % movable_patch.group_id)
            xfft = self.analyzer.get_xfft(self.fft_size)
            
            plt.plot(
                xfft,
                avg_fft,
                alpha=.15)
            #plt.show()
        pass
    '''

class DebugPatchAnalyzerV2(DebugPatchAnalyzer):
    def __init__(self, json_path):
        super().__init__(json_path)
        
        file_name, ext = os.path.splitext(os.path.basename(json_path))
        self.vid_path = '../videos/%s.mp4' % file_name
        
        self._reload_vid_reader()
        
        self.vid_display = VideoDisplay(
            'foo',
            self.vid_reader.info.width,
            self.vid_reader.info.height,
            0.5)
    
    def load_frame(self, frame_num):
        if frame_num < self.vid_frame_num:
            self._reload_vid_reader()
        
        while self.vid_frame_num < frame_num:
            self._load_next_frame()
    
    def render_frame(self):
        self._show_frame()
    
    def draw_point(self, point_id, x, y):
        x = int(round(x))
        y = int(round(y))
        cv.circle(
            self.vid_frame,
            (x, y),
            5,
            (255, 255, 255),
            -1)
        cv.putText(self.vid_frame,
            str(point_id),
            (x, y),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv.LINE_AA)
    
    def _show_frame(self):
        self.vid_display.show(self.vid_frame)
        cv.waitKey(150)
    
    def _reload_vid_reader(self):
        self.vid_reader = VideoFileReader(self.vid_path)
        self.vid_frame = None
        self.vid_frame_num = 0

    def _load_next_frame(self):
        frame = self.vid_reader.get_next_frame()
        if frame is None:
            raise Exception('Could not load next frame')
        self.vid_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.vid_frame_num += 1









