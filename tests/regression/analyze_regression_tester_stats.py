from pet_breathy.video_display import VideoDisplay
from pet_breathy.video_reader import VideoReader
from pet_breathy import video_reader
from pet_breathy.point import Point
from pet_breathy.point import calc_point_dist
from pet_breathy.patch_cluster import PatchClusterManager, PatchClusterFrame

import sys
import json
import enum
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp

def main():
    args = sys.argv[1:]
    if len(args) == 1:
        run(*args)
    else:
        print('Usage: analyze_regression_tester_stats.py <stats .json path>')

def run(stats_json_path):
    analyzer = Analyzer(stats_json_path)
    analyzer.run()

class Analyzer:
    def __init__(self, stats_json_path):
        self.stats_json_path = stats_json_path
        
        with open(self.stats_json_path) as fin:
            self.stats_json = json.load(fin)
        
        # dict_keys(['info', 'decimation', 'patch_stats_lookup', 'top_patch_ids'])
        #print(self.stats_json.keys())
        print('>>> len', len(self.stats_json['patch_stats_lookup']))
        
        self.vid_info = self.stats_json['info']
        
        print('vid info:', self.vid_info)
        # vid info: {'fps': 30, 'width': 1920, 'height': 1080, 'frame_count': 570, 'video_path': '../videos/Blaze2.mp4', 'start_frame_num': 0, 'stop_frame_num': 570}

        self.display_ratio = 0.5
        
        self.normal_display_enabled = False
        self.decimated_display_enabled = False
        self.decimated_plots_enabled = False
        self.deicmated_debug_inspect_enabled = False
        self.patch_cluster_enabled = True
        self.patch_cluster_display_enabled = False
        
        if self.normal_display_enabled:
            self.normal_display = VideoDisplay(
                'Debug - Normal playback',
                int(self.vid_info['width']),
                int(self.vid_info['height']),
                self.display_ratio)
        
        if self.decimated_display_enabled or self.deicmated_debug_inspect_enabled or self.patch_cluster_display_enabled:
            self.decimated_display = VideoDisplay(
                'Debug - Decimated playback',
                int(self.vid_info['width']),
                int(self.vid_info['height']),
                self.display_ratio)

        self.vid_reader = video_reader.create_video_file_reader(self.vid_info['video_path'])
        self.decimation = int(self.stats_json['decimation'])
        self.frame_count = int(self.vid_info['frame_count'])
        self.pause_before_exiting = False
        self.fps = int(self.vid_info['fps'])
        self.start_frame_num = int(self.vid_info['start_frame_num'])
        self.stop_frame_num = int(self.vid_info['stop_frame_num'])
        self.vid_width = int(self.vid_info['width'])
        self.vid_height = int(self.vid_info['height'])
        
        self.radius = int(round(.15 * min(self.vid_width, self.vid_height)))
        print('>>> radius:', self.radius)
        self.bpm_threshold = 3
        self.cluster_manager = PatchClusterManager(self.radius, self.bpm_threshold)
        self._load_cluster_manager()
        
        #print('top_patch_ids', len(self.stats_json['top_patch_ids']))
        #for top_patch_ids in self.stats_json['top_patch_ids']:
        #    print(' -', top_patch_ids)
        #     - {'frame_num': 204, 'patch_ids': [9, 8, 22, 23, 20, 30, 6, 19, 11, 25]}
        #print('patch_stats_lookup', len(self.stats_json['patch_stats_lookup']))
        # top_patch_ids 95
        # patch_stats_lookup 59
    
    def run(self):
        if self.decimated_plots_enabled:
            self._run_plot_analysis()
        
        if self.normal_display_enabled or self.decimated_display_enabled:
            self._run_video_analysis()
        
        if self.deicmated_debug_inspect_enabled:
            self._run_debug_inspect()
        
        if self.patch_cluster_enabled:
            self._run_patch_cluster_analysis()
        
        if self.patch_cluster_display_enabled:
            self._run_patch_cluster_video_analysis()
    
    def _run_patch_cluster_video_analysis(self):
        for self.vid_frame_num in range(self.frame_count):
            #print('Frame: ', self.vid_frame_num)
            
            frame_stop = 138
            frame_stop_enable = False
            if frame_stop_enable and self.vid_frame_num != frame_stop:
                continue
            
            self.vid_frame = self.vid_reader.get_next_frame()
            
            if self.vid_frame_num < self.start_frame_num:
                continue
            
            if self.vid_frame_num % self.decimation == 0:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> frame_num: %s' % self.vid_frame_num)
                
                self._process_cluster_frame()
                
                self.decimated_display.show(self.decimated_vid_frame)
            
            if frame_stop_enable and self.vid_frame_num == frame_stop:
                self.pause_before_exiting = True
                break
            
            exit_status = wait_key_press(self.fps)
            
            if exit_status != ExitStatus.DoNothing:
                if exit_status == ExitStatus.PauseAndExit:
                    self.pause_before_exiting = True
                break
        
        if self.pause_before_exiting:
            while wait_key_press_with_timeout(10) == ExitStatus.DoNothing:
                pass
    
    def _process_cluster_frame(self):
        self.decimated_vid_frame = cv.cvtColor(self.vid_frame, cv.COLOR_BGR2GRAY)
        
        '''
        find all the clusters that are active for that frame
        '''
        def convert_float_point(point):
            return Point(
                int(round(point.x)),
                int(round(point.y)))
        
        def draw_cluster(cluster):
            #if cluster.get_bpm() < 60:
            #    return
            try:
                frames = cluster.get_frames(self.vid_frame_num)
            except KeyError:
                return
            top_frame = frames[0]
            center_point = convert_float_point(top_frame.center)
            line_width = len(frames)
            frame_avg = cluster.get_frame_average(self.vid_frame_num)
            
            patch_str_items = [
                f' {cluster.get_id()}',
                f'- {round(frame_avg.bpm_avg, 5)}',
                f'- {round(frame_avg.bpm_strength_avg, 5)}'
            ]
            
            cv.circle(
                self.decimated_vid_frame,
                (center_point.x, center_point.y),
                5,
                (255, 255, 255),
                cv.FILLED)
            
            '''
            cv.circle(
                self.decimated_vid_frame,           # img
                (center_point.x, center_point.y),   # center
                self.radius,                        # radius
                (255, 255, 255),                    # color
                line_width,                         # thickness
                cv.LINE_AA)                         # lineType
            '''
            
            for i, patch_str_item in enumerate(patch_str_items):
                cv.putText(self.decimated_vid_frame,
                    patch_str_item,
                    (center_point.x, center_point.y + ((i + 1) * 15)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    .5,
                    (255, 255, 255),
                    2,
                    cv.LINE_AA)
            
            for i in range(1, len(frames)):
                frame = frames[i]
                point = convert_float_point(frame.center)
                cv.circle(
                    self.decimated_vid_frame,
                    (point.x, point.y),
                    5,
                    (255, 255, 255),
                    cv.FILLED)
                
                cv.line(
                    self.decimated_vid_frame,
                    (point.x, point.y),
                    (center_point.x, center_point.y),
                    (255, 255, 255),
                    1,
                    cv.LINE_AA)
        
        for group_id, group in self.cluster_manager.all_groups.items():
            if not group.frame_num_in_range(self.vid_frame_num):
                continue
            
            all_cluster_ids = list(group.all_cluster_ids)
            
            for i in range(len(all_cluster_ids)):
                cluster_id_i = all_cluster_ids[i]
                cluster_i = self.cluster_manager.all_clusters[cluster_id_i]
                try:
                    cluster_center_i = cluster_i.get_center(self.vid_frame_num)
                except KeyError:
                    continue
                
                draw_cluster(cluster_i)
                
                p0 = convert_float_point(cluster_center_i)
                
                for j in range(i, len(all_cluster_ids)):
                    cluster_id_j = all_cluster_ids[j]
                    cluster_j = self.cluster_manager.all_clusters[cluster_id_j]
                    try:
                        cluster_center_j = cluster_j.get_center(self.vid_frame_num)
                    except KeyError:
                        continue
                    
                    p1 = convert_float_point(cluster_center_j)
                    
                    cv.line(
                        self.decimated_vid_frame,
                        (p0.x, p0.y),
                        (p1.x, p1.y),
                        (255, 255, 255),
                        1,
                        cv.LINE_AA)
        
        cv.putText(self.decimated_vid_frame,
            'Frame %s' % self.vid_frame_num,
            (0,25),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv.LINE_AA)
    
    def _run_patch_cluster_analysis(self):
        for cluster_id, cluster in self.cluster_manager.all_clusters.items():
            summary = cluster.get_summary()
            print(cluster_id, summary)
        
        print(f'Clusters: {len(self.cluster_manager.all_clusters)}')
        
        #fig, axs = plt.subplots(nrows=3, ncols=2)
        fig, axs = plt.subplot_mosaic(
            [['1', '2', '3'],
            ['4', '5', '3'],
            ['6', '7', '3']])
        
        axs = [
            [axs['1'], axs['2'], axs['3']],
            [axs['4'], axs['5'], axs['3']],
            [axs['6'], axs['7'], axs['3']]
            ]
        
        # 'width': 1920, 'height': 1080,
        axs[1][0].set_xlim([0, self.vid_width])
        axs[1][0].set_ylim([0, self.vid_height])
        axs[2][0].set_xlim([0, self.vid_width])
        axs[2][0].set_ylim([0, self.vid_height])
        
        img = self.vid_reader.get_next_frame()
        axs[1][0].imshow(img)
        axs[2][0].imshow(img)
        
        def get_line_width(num_frames):
            return num_frames / 500.0
        
        centers_x = []
        centers_y = []
        centers_size = []
        for i in range(10):
            centers_x.append([])
            centers_y.append([])
            centers_size.append([])
        
        center_x = []
        center_y = []
        center_sizes = []
        
        all_bpms = []
        all_bpm_strengths = []
        
        for cluster_id, cluster in self.cluster_manager.all_clusters.items():
            summary = cluster.get_summary()
            line_width = get_line_width(summary.num_frames)
            if line_width == 0:
                continue
            x = np.arange(summary.begin_frame_num, summary.end_frame_num + 1)
            y = np.full(summary.num_frames, summary.bpm_avg)
            axs[0][0].plot(x, y, linewidth=line_width)
            
            #all_bpms.append(summary.bpm_avg)
            #all_bpm_strengths.append(summary.bpm_strength_avg)
            
            x = cluster.curr_center.x
            y = cluster.curr_center.y
            s = .5 * summary.num_frames
            
            if cluster.bpm <= 10:
                centers_x[0].append(x)
                centers_y[0].append(y)
                centers_size[0].append(s)
            elif cluster.bpm <= 20:
                centers_x[1].append(x)
                centers_y[1].append(y)
                centers_size[1].append(s)
            elif cluster.bpm <= 30:
                centers_x[2].append(x)
                centers_y[2].append(y)
                centers_size[2].append(s)
            elif cluster.bpm <= 40:
                centers_x[3].append(x)
                centers_y[3].append(y)
                centers_size[3].append(s)
            elif cluster.bpm <= 50:
                centers_x[4].append(x)
                centers_y[4].append(y)
                centers_size[4].append(s)
            elif cluster.bpm <= 60:
                centers_x[5].append(x)
                centers_y[5].append(y)
                centers_size[5].append(s)
            elif cluster.bpm <= 70:
                centers_x[6].append(x)
                centers_y[6].append(y)
                centers_size[6].append(s)
            elif cluster.bpm <= 80:
                centers_x[7].append(x)
                centers_y[7].append(y)
                centers_size[7].append(s)
            elif cluster.bpm <= 90:
                centers_x[8].append(x)
                centers_y[8].append(y)
                centers_size[8].append(s)
            else:
                centers_x[9].append(x)
                centers_y[9].append(y)
                centers_size[9].append(s)
            
            center_x.append(x)
            center_y.append(y)
            center_sizes.append(s)
        
        for i in range(10):
            if centers_x[i]:
                axs[1][0].scatter(centers_x[i], centers_y[i], centers_size[i], label=f'{(i+1)*10}')
        
        #axs[2][1].hist(all_bpms, bins=100, weights=all_bpm_strengths)
        
        #__LEFT_OFF_HERE__
        '''
        - Looking at the trend of bpm, number of frames ... probably should look at strength instead.
        - Explore start end and of cluster
        - Need to keep looking for a way to track this while bpm evolves if there isn't a clear winner
        - Looking at individual clusters above
        - Could look at cluster groups as well.
        '''
        
        #axs[1].scatter(center_x, center_y, s=center_sizes)
        #count = 0
        #for cluster_id, cluster in self.cluster_manager.all_clusters.items():
        #    axs[1].annotate(f'{cluster.bpm}', (center_x[count], center_y[count]))
        #    count += 1
        
        groups = list(self.cluster_manager.all_groups.values())
        #groups.sort(key=lambda x: x.bpm_strength_max, reverse=True)
        groups.sort(key=lambda x: x.get_score(), reverse=True)
        
        bpm_heat_map = [ ]
        
        print('>>>> num groups:', len(self.cluster_manager.all_groups))
        #for group_id, group in self.cluster_manager.all_groups.items():
        for group in groups[:5]:
            bpm_avgs = [ ]
            bpm_strengths = [ ]
            for cluster_id in group.all_cluster_ids:
                cluster = self.cluster_manager.all_clusters[cluster_id]
                summary = cluster.get_summary()
                bpm_avgs.append(summary.bpm_avg)
                bpm_strengths.append(summary.bpm_strength_avg)
            bpm_avg = np.average(bpm_avgs, weights=bpm_strengths)
            bpm_mode = sp.stats.mode(bpm_avgs)[0]
            
            print(f'- group id: {group.group_id}, '
                f'bpm avg: {round(bpm_avg,2)} mode: {round(bpm_mode,2)}, bpm: {round(group.bpm,2)} '
                f'bpm range: [{round(group.bpm_min,2)} {round(group.bpm_max,2)}], '
                f'frames: {group.end_frame_num - group.begin_frame_num}, '
                f'bframe: {group.begin_frame_num}, eframe: {group.end_frame_num}, '
                f'strength: {round(group.bpm_strength_max,5)}')
            
            info = sp.stats.describe(bpm_avgs)
            print(f'  - mean: {info.mean}, var: {info.variance}, skew: {info.skewness}, kurtosis: {info.kurtosis}')
            centers_x = [ ]
            centers_y = [ ]
            for cluster_id in group.all_cluster_ids:
                cluster = self.cluster_manager.all_clusters[cluster_id]
                x = cluster.curr_center.x
                y = cluster.curr_center.y
                centers_x.append(x)
                centers_y.append(y)
                
                #summary = cluster.get_summary()
                #all_bpms.append(summary.bpm_avg)
                #all_bpm_strengths.append(summary.bpm_strength_avg)
            
            #all_bpms = []
            #all_bpm_strengths = []
            #bottom = None
            
            # Get average bpm for each frame
            x = np.arange(self.frame_count)
            y = np.zeros(self.frame_count)
            for frame_num in range(self.frame_count):
                if not group.frame_num_in_range(frame_num):
                    continue
                
                bpm_avg_total = 0
                bpm_avg_total_count = 0
                
                for cluster_id in group.all_cluster_ids:
                    cluster = self.cluster_manager.all_clusters[cluster_id]
                    if not cluster.frame_num_in_range(frame_num):
                        continue
                    try:
                        frame_avg = cluster.get_frame_average(frame_num)
                    except KeyError:
                        continue
                    
                    bpm_avg_total += frame_avg.bpm_avg
                    bpm_avg_total_count += 1
                    
                    all_bpms.append(frame_avg.bpm_avg)
                    all_bpm_strengths.append(frame_avg.bpm_strength_avg)
                
                if bpm_avg_total_count:
                    y[frame_num] = bpm_avg_total / bpm_avg_total_count
            
            #if bottom is None:
            #    bottom = np.ndarray(len(all_bpm_strengths))
            
            num_bins = 50
            #axs[2][1].hist(all_bpms, bins=num_bins, histtype='bar', stacked=True, label=f'{round(group.bpm,2)}')
            #axs[2][1].bar(all_bpms, all_bpm_strengths, 0.5, bottom=bottom, label=f'{round(group.bpm,2)}')
            #axs[2][1].bar(all_bpms, all_bpm_strengths, 0.5, label=f'{round(group.bpm,2)}')
            #bottom += all_bpm_strengths
            
            axs[1][1].scatter(x, y, label=f'{round(bpm_avg,2)}')
            
            axs[2][0].scatter(centers_x, centers_y, label=f'{round(group.bpm,2)}')
            #axs[0][1].plot(np.arange(len(bpm_avgs)), bpm_avgs, label=f'{round(bpm_avg,2)}')
        
        bpm_heat_map = [ ]
        bpm_heat_map_step = 10
        for frame_num in range(0, self.frame_count, bpm_heat_map_step):
            bpm_heat_map.append([ 0 ] * 100)
        
        for frame_num in range(self.frame_count):
            for group in groups[:5]:
                if not group.frame_num_in_range(frame_num):
                    continue
                
                for cluster_id in group.all_cluster_ids:
                    cluster = self.cluster_manager.all_clusters[cluster_id]
                    if not cluster.frame_num_in_range(frame_num):
                        continue
                    try:
                        frame_avg = cluster.get_frame_average(frame_num)
                    except KeyError:
                        continue
                    
                    idx = frame_num // bpm_heat_map_step
                    bpm_heat_map[idx][int(round(frame_avg.bpm_avg))] += 1
        
        num_bins = 100
        #if all_bpm_strengths:
        #    axs[2][1].hist(all_bpms, bins=num_bins, weights=all_bpm_strengths)
        #else:
        axs[2][1].hist(all_bpms, bins=num_bins, range=(1,100))
        
        axs[0][2].imshow(bpm_heat_map, interpolation='nearest')
        
        axs[1][0].legend()
        axs[2][0].legend()
        #axs[0][1].legend()
        #axs[1][1].legend()
        #axs[2][1].legend()
        
        plt.show()
    
    def _load_cluster_manager(self):
        patch_stats_lookup = self.stats_json['patch_stats_lookup']
        top_patch_ids_lookup = self.stats_json['top_patch_ids']
        
        # dict_keys(['0', '6', '12', '18', '24', ...])
        # print(top_patch_ids_lookup.keys())
        
        for frame_num in range(self.start_frame_num, self.stop_frame_num, self.decimation):
            # Adding them all
            # Adding only top patch ids
            try:
                top_patch_ids = top_patch_ids_lookup[str(frame_num)]
            except KeyError:
                continue
            
            for patch_id, patch in patch_stats_lookup.items():
                if not int(patch_id) in top_patch_ids:
                    continue
                # dict_keys(['patch_id', 'patch_type', 'frames'])
                # print(patch.keys())
                frames = patch['frames']
                # {'point': ['287.14282', '158.65073'], 'state': 'None', 'seg_length': None, 'signal': {}, 'signal_score': {}}
                # print(frames[str(frame_num)])
                frame = frames[str(frame_num)]
                signal = frame['signal']
                if signal:
                    # {'strength': '0.006365602223179484', 'bpm_est': '42.857142857142854', 'bpm_precision': '18.75', 'fft_size': 16, 'fft_level': 0, 'decimation': 6}
                    # print(signal)
                    point = frame['point']
                    
                    self.cluster_manager.add_frame(
                        PatchClusterFrame(
                            Point(float(point[0]), float(point[1])),
                            float(signal['bpm_est']),
                            float(signal['strength']),
                            frame_num))
            
            self.cluster_manager.finished_adding_frames_for_frame_num()
        
        '''
        - Can I stich a timeline here?
        - How to show options to user,
            - Like top 3 cluster groups and their runtimes
        '''
        print_enabled = True
        if print_enabled:
            for cluster_id, cluster in self.cluster_manager.all_clusters.items():
                runtime = (cluster.end_frame_num - cluster.begin_frame_num) / self.fps
                if runtime >= 5.0:
                    start = cluster.begin_frame_num / self.fps
                    end = cluster.end_frame_num / self.fps
                    summary = cluster.get_summary()
                    print(f'- Cluster {cluster_id}, runtime: {runtime}, start: {start}, end: {end}, summary: {summary}')
            
            for group_id, group in self.cluster_manager.all_groups.items():
                runtime = (group.end_frame_num - group.begin_frame_num) / self.fps
                if runtime >= 1.0:
                    start = group.begin_frame_num / self.fps
                    end = group.end_frame_num / self.fps
                    print(f'- Group {group_id}, '
                        f'clusters: {len(group.all_cluster_ids)}, '
                        f'runtime: {runtime}, start: {start}, end: {end}, '
                        f'bpm: {round(group.bpm, 2)}, '
                        f'bpm range: [{round(group.bpm_min, 2)} {round(group.bpm_max, 2)}], '
                        f'bpm strength: {round(group.bpm_strength_max, 5)}')
                    
                    for cluster_id in group.all_cluster_ids:
                        cluster = self.cluster_manager.all_clusters[cluster_id]
                        start = cluster.begin_frame_num / self.fps
                        end = cluster.end_frame_num / self.fps
                        summary = cluster.get_summary()
                        runtime = (cluster.end_frame_num - cluster.begin_frame_num) / self.fps
                        print(f'  - Cluster {cluster_id}, runtime: {runtime}, start: {start}, end: {end}, summary: {summary}')
                    
                    #print(f'- Group {group_id}, runtime: {runtime}, start: {start}, end: {end}')
                    #print(f'  - Cluster ids: {", ".join(str(cid) for cid in group.all_cluster_ids)}')
                    #print(f'  - bpm: {group.bpm}, bpm range: [{group.bpm_min} {group.bpm_max}], bpm strength: {group.bpm_strength_max}')
        
        #__TODO__
        '''
        Not sure if this is working, look at blaze6 video, there's a mix of 18 bpm and 32 bpms, looks poopy, not sure if I noticed that before.
        
        now groups are output with runtime with start and end time
        need to look at the clusters to get average bpm and strength, then report that per group
        the top groups should then be presented to the user to select which one is the one to use
        
        Still need a visual time line of sorts so the user can see where the bpm data originated from
        
        For example, a begin and end frame for that group, or a stretch would be like a slider bar, think Chrome gui slider thing that lets you see what the screen shot was at a particular time, except this is showing you where the bpm data was measured over time
        '''
        
        print(f'Total runtime: {self.frame_count / self.fps}')
    
    def _run_debug_inspect(self):
        for self.vid_frame_num in range(self.frame_count):
            self.vid_frame = self.vid_reader.get_next_frame()
            
            if self.vid_frame_num < self.start_frame_num:
                continue
            
            if self.vid_frame_num % self.decimation == 0:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> frame_num: %s' % self.vid_frame_num)
                
                self._debug_process_frame()
                
                self.decimated_display.show(self.decimated_vid_frame)
            
            exit_status = wait_key_press(self.fps)
            
            if exit_status != ExitStatus.DoNothing:
                if exit_status == ExitStatus.PauseAndExit:
                    self.pause_before_exiting = True
                break
        
        if self.pause_before_exiting:
            while wait_key_press_with_timeout(10) == ExitStatus.DoNothing:
                pass
    
    def _debug_process_frame(self):
        self.decimated_vid_frame = cv.cvtColor(self.vid_frame, cv.COLOR_BGR2GRAY)
        
        patch_stats_lookup = self.stats_json['patch_stats_lookup']
        top_patch_ids_lookup = self.stats_json['top_patch_ids']
        
        # top_patch_ids_lookup - Keys: frame id, Values: list of top patch ids
        top_patch_ids = top_patch_ids_lookup[str(self.vid_frame_num)]
        print(top_patch_ids)
        
        # 20231003_162545_sophie.json 
        #patch_id = 8   # good point
        patch_ids = list(range(4, len(patch_stats_lookup) - 1))

        for patch_id in patch_ids:
            patch = patch_stats_lookup[str(patch_id)]
            patch_frames = patch['frames']
            patch_frame = patch_frames[str(self.vid_frame_num)]
            patch_point = patch_frame['point']
            patch_signal = patch_frame['signal']
            
            print('pid:', patch_id, patch_frame)
            
            if patch_id in top_patch_ids: 
                patch_str = f'>>{patch_id}<<'
            else:
                patch_str = f'{patch_id}'
            
            patch_str_items = [ ]
            if patch_signal:
                bpm_est = patch_signal["bpm_est"]
                strength = round(float(patch_signal["strength"]), 5)
                patch_str_items.append(f'- {bpm_est}')
                patch_str_items.append(f'- {strength}')
            
            point = Point(
                int(float(patch_point[0])),
                int(float(patch_point[1])))
            
            cv.circle(
                self.decimated_vid_frame,
                (point.x, point.y),
                5,
                (255, 255, 255),
                cv.FILLED)
            
            cv.putText(self.decimated_vid_frame,
                patch_str,
                (point.x + 2, point.y),
                cv.FONT_HERSHEY_SIMPLEX,
                .5,
                (255, 255, 255),
                2,
                cv.LINE_AA)
            
            cv.putText(self.decimated_vid_frame,
                'Frame %s' % self.vid_frame_num,
                (0,25),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv.LINE_AA)
    
    def _run_plot_analysis(self):
        # self.stats_json.keys()
        # dict_keys(['info', 'decimation', 'patch_stats_lookup', 'top_patch_ids
        
        patch_stats_lookup = self.stats_json['patch_stats_lookup']
        top_patch_ids = self.stats_json['top_patch_ids']
        
        # 95 <class 'dict'> dict_keys(['0', '6', '12',
        #print(len(top_patch_ids), type(top_patch_ids), top_patch_ids.keys())
        
        # Keys:     Patch IDs   ['0', '1', '2', '3', '4', ... ]
        # Values:   
        #print(type(patch_stats_lookup), patch_stats_lookup.keys())
        #print(len(patch_stats_lookup))
        
        # patch
        #print(type(patch_stats_lookup['0']), patch_stats_lookup['0'].keys())
        # dict_keys(['patch_id', 'patch_type', 'frames'])
        
        # Num frames 154
        #print('num frames:', len(patch_stats_lookup['0']['frames']))
        
        fig, axs = plt.subplots(nrows=3, ncols=2)
        
        # Key:      Patch ID
        # Value:    List of points
        bpms_dict = { }
        sig_strengths_dict = { }
        displacements_dict = { }
        
        print(f'>>> Number of patches: {len(patch_stats_lookup)}')
        
        for patch_id in range(len(patch_stats_lookup)):
            patch = patch_stats_lookup[str(patch_id)]
            frames = patch['frames']
            
            bpms = [ ]
            sig_strengths = [ ]
            displacements = [ ]
            
            prev_point = None
            
            #print(type(frames))
            # Key:      Frame number
            # Value:    
            #print(frames.keys())
            
            # <class 'dict'> dict_keys(['point', 'state', 'seg_length', 'signal', 'signal_score'])
            #print(type(frames['0']), frames['0'].keys())
            
            #print('len(frames):', len(frames))
            for frame_num, frame in frames.items():
                frame_num = int(frame_num)
                #print(frame_num, frame)
                
                # frame
                # {'point': ['288.21024', '162.20981'], 'state': 'None', 'seg_length': None, 'signal': {}, 'signal_score': {}, 'y_avg_pts': ['162.1049041748047', '162.1049041748047'], 'y_act_pts': ['162.0', '162.20980834960938'], 'spectra': []}
                x = float(frame['point'][0])
                y = float(frame['point'][1])
                state = frame['state']
                seg_length = frame['seg_length']
                signal = frame['signal']
                signal_score = frame['signal_score']
                y_avg_pts = np.array(frame['y_avg_pts'], dtype=float)
                y_act_pts = np.array(frame['y_act_pts'], dtype=float)
                spectra = np.array(frame['spectra'], dtype=float)
                
                p = Point(x, y)
                if prev_point:
                    displacement = calc_point_dist(p, prev_point)
                else:
                    displacement = 0
                prev_point = p
                top_patches = top_patch_ids[str(frame_num)][:10]
                
                # if signal:
                    # print('>>>', signal)
                    # >>> {'strength': '0.023820822646710024', 'bpm_est': '60.0', 'bpm_precision': '18.75', 'fft_size': 16, 'fft_level': 0, 'decimation': 6}
                
                use_option_1 = False
                use_top_patches = True
                
                added = False
                
                # PXL_20230824_035440033 (foi 162, 306)
                poi_0 = 9   # On Wagner
                poi_1 = 10  # On bookshelf
                # PXL_20230825_040038487
                #poi_0 = 45  # On cubby/bookshelf
                #poi_1 = 38  # On Wagner
                
                foi = 300 // self.decimation *  self.decimation
                sig_strength = 0
                
                if use_option_1:
                    is_poi = patch_id == poi_0 or patch_id == poi_1
                    
                    if patch_id < 4 and is_poi:
                        added = True
                    elif signal and is_poi:
                        sig_strength = float(signal['strength'])
                        bpm = float(signal['bpm_est'])
                        added = True
                        
                        #print(f'fn: {frame_num}, patch_id: {patch_id}, displacement: {displacement}, bpm: {bpm}, state: {state}')
                        
                    if added and frame_num == foi and is_poi:
                        axs[0][1].plot(np.arange(len(y_avg_pts)), y_avg_pts, label=f'Avg {patch_id}')
                        axs[0][1].plot(np.arange(len(y_avg_pts)), y_act_pts, label=f'Act {patch_id}')
                        axs[1][1].plot(np.arange(len(y_avg_pts)), y_act_pts - y_avg_pts, label=f'{patch_id}')
                        axs[2][1].plot(np.arange(len(spectra)), spectra, label=f'{patch_id}')
                
                if use_top_patches:
                    patch_id_in_top_patches = patch_id in top_patches
                    #if signal and (patch_id == top_patches[0] or patch_id == top_patches[1]):
                    if signal and patch_id_in_top_patches:
                        sig_strength = float(signal['strength'])
                        bpm = float(signal['bpm_est'])
                    else:
                        sig_strength = 0
                        bpm = 0
                
                if (not use_option_1 and patch_id_in_top_patches) or (use_option_1 and added and sig_strength):
                    sig_strengths.append(sig_strength)
                    bpms.append(bpm)
                    displacements.append(displacement)
            
            if bpms and sig_strengths and displacements:
                bpms_dict[patch_id] = bpms
                sig_strengths_dict[patch_id] = sig_strengths
                displacements_dict[patch_id] = displacements
                #break
            #break
        
        for patch_id in bpms_dict.keys():
            bpms = bpms_dict[patch_id]
            sig_strengths = sig_strengths_dict[patch_id]
            displacements = displacements_dict[patch_id]
            
            axs[0][0].plot(np.arange(len(sig_strengths)), sig_strengths, label=f'{patch_id}')
            axs[1][0].scatter(np.arange(len(bpms)), bpms, label=f'{patch_id}')
            axs[2][0].plot(np.arange(len(displacements)), displacements, label=f'{patch_id}')
        
        if not use_top_patches:
            axs[0][0].legend()
            axs[1][0].legend()
            axs[2][0].legend()
            axs[0][1].legend()
            axs[1][1].legend()
            axs[2][1].legend()
        plt.show()
    
    def _run_video_analysis(self):
        for self.vid_frame_num in range(self.frame_count):
            #print('Frame: ', self.vid_frame_num)
            
            self.vid_frame = self.vid_reader.get_next_frame()
            
            if self.vid_frame_num < self.start_frame_num:
                continue
            
            if self.normal_display_enabled:
                self.normal_display.show(self.vid_frame)

            if self.vid_frame_num % self.decimation == 0:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> frame_num: %s' % self.vid_frame_num)
                
                if self.decimated_display_enabled:
                    self._process_frame()
                    
                    self.decimated_display.show(self.decimated_vid_frame)
            
            exit_status = wait_key_press(self.fps)
            
            if exit_status != ExitStatus.DoNothing:
                if exit_status == ExitStatus.PauseAndExit:
                    self.pause_before_exiting = True
                break
        
        if self.pause_before_exiting:
            while wait_key_press_with_timeout(10) == ExitStatus.DoNothing:
                pass

    def _process_frame(self):
        self.decimated_vid_frame = cv.cvtColor(self.vid_frame, cv.COLOR_BGR2GRAY)
        '''
        print('top_patch_ids', len(self.stats_json['top_patch_ids']))
        for top_patch_ids in self.stats_json['top_patch_ids']:
            print(' -', top_patch_ids)
        print('patch_stats_lookup', len(self.stats_json['patch_stats_lookup']))
        '''
        top_patch_ids_lookup = self.stats_json['top_patch_ids']
        top_patch_ids = top_patch_ids_lookup[str(self.vid_frame_num)]
        
        patch_stats_lookup = self.stats_json['patch_stats_lookup']
        
        for patch_idx, patch_id in enumerate(top_patch_ids):
            patch = patch_stats_lookup[str(patch_id)]
            
            #dict_keys(['patch_id', 'patch_type', 'frames'])
            #print(patch.keys())
            
            patch_frames = patch['frames']
            patch_frame = patch_frames[str(self.vid_frame_num)]
            patch_point = patch_frame['point']
            patch_signal = patch_frame['signal']
            
            patch_str = f'{patch_id}'
            patch_str_items = [ ]
            if patch_signal:
                bpm_est = patch_signal["bpm_est"]
                strength = round(float(patch_signal["strength"]), 5)
                patch_str_items.append(f'- {bpm_est}')
                patch_str_items.append(f'- {strength}')
            
            point = Point(
                int(float(patch_point[0])),
                int(float(patch_point[1])))
            
            #print('>>> patch id:', patch_id, f'{point.x} {point.y}')
            
            for other_patch_id in top_patch_ids:
                if patch_id == other_patch_id:
                    continue
                other_patch = patch_stats_lookup[str(other_patch_id)]
                other_patch_frames = other_patch['frames']
                other_patch_frame = other_patch_frames[str(self.vid_frame_num)]
                other_patch_point = other_patch_frame['point']
                other_point = Point(
                    int(float(other_patch_point[0])),
                    int(float(other_patch_point[1])))
                
                patch_dist = calc_point_dist(point, other_point)
                
                #print('    - Patch id:', other_patch_id, patch_dist)
            
            cv.circle(
                self.decimated_vid_frame,
                (point.x, point.y),
                5,
                (255, 255, 255),
                cv.FILLED)
            
            if patch_idx < 3:
                cv.circle(
                    self.decimated_vid_frame,   # img
                    (point.x, point.y),         # center
                    100,                        # radius
                    (255, 255, 255),            # color
                    1,                          # thickness
                    cv.LINE_AA)                 # lineType
            
            cv.putText(self.decimated_vid_frame,
                patch_str,
                (point.x + 2, point.y),
                cv.FONT_HERSHEY_SIMPLEX,
                .5,
                (255, 255, 255),
                2,
                cv.LINE_AA)
            
            for i, patch_str_item in enumerate(patch_str_items):
                cv.putText(self.decimated_vid_frame,
                    patch_str_item,
                    (point.x, point.y + ((i + 1) * 15)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    .5,
                    (255, 255, 255),
                    2,
                    cv.LINE_AA)
        
        cv.putText(self.decimated_vid_frame,
            'Frame %s' % self.vid_frame_num,
            (0,25),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv.LINE_AA)

def wait_key_press(fps):
    wait_time_ms = int(1.0 / fps * 1000)
    return wait_key_press_with_timeout(wait_time_ms)

def wait_key_press_with_timeout(wait_time_ms):
    k = cv.waitKey(wait_time_ms)
    if k == 99: # 'c'
        exit_status = ExitStatus.JustExit
    elif k == 115: # 's'
        exit_status = ExitStatus.PauseAndExit
    else:
        exit_status = ExitStatus.DoNothing
    return exit_status

class ExitStatus(enum.Enum):
    DoNothing = enum.auto()
    JustExit = enum.auto()
    PauseAndExit = enum.auto()


if __name__ == '__main__':
    main()
    #test_cluster_frame()
