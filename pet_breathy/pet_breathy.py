from pet_breathy.video_info import VideoInfo
from pet_breathy.optical_flow import OpticalFlow
from pet_breathy.signal_analyzer import SignalAnalyzer
from pet_breathy import signal_analyzer
from pet_breathy.patch_analyzer import PatchAnalyzer
from pet_breathy.point_group_manager import PointGroupManager
from pet_breathy.point_monitor import PointMonitor
from pet_breathy.point import Point
from pet_breathy.segment import Segment
from pet_breathy import signal
from pet_breathy.prng_point_gen import PrngPointGen
from pet_breathy import movable_patch
from pet_breathy.stats import Stats
from pet_breathy.patch_stats import AnalysisState, PatchFrameStats

from pet_breathy.static_patch import StaticPatch
from pet_breathy.circle_patch import CirclePatch

import numpy as np
import cv2 as cv
import functools

class PetBreathy:
    def __init__(self, info: VideoInfo, max_points: int, decimation: int, prev_frame: np.ndarray, debug = False):
        self.info = info
        self.max_points = max_points
        self.decimation = decimation
        self.prev_frame = self._preprocess_frame(prev_frame)
        
        self.debug = debug
        self.debug_prints = False
        self.enable_debug_points = False
        
        self._setup()

    def get_max_runtime_per_frame(self) -> float:
        return self.decimation / self.info.fps

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if (self.frame_count % self.decimation) == 0:
            self._process_frame(frame)
        
        self.frame_count += 1
        
        # TODO May want to draw on this frame what we're seeing, like BPM,
        # where in the image BPM was calculated, etc ...
        return frame

    def done(self):
        self.patch_analyzer.done()
        
        if self.debug:
            self.debug_thing.done()

    def set_debug_prints(self, enabled: bool):
        self.debug_prints = enabled
        self.patch_analyzer.set_debug_prints(enabled)
        
        for patch in self.patch_lookup.values():
            patch.set_debug_prints(enabled)

    def get_stats(self) -> Stats:
        return self.stats

    def _process_frame(self, frame: np.ndarray):
        frame = self._preprocess_frame(frame)
        
        old_points = self.manager.get_points()
        new_points, _, _ = \
            self.flow.calc(self.prev_frame, frame, old_points)
        
        self.prev_frame = frame
        
        self.monitor.check_for_bad_points(old_points, new_points)
        
        self.manager.update_points(new_points, self.monitor)
        
        # For each patch, add points to segments
        for patch in self.patch_lookup.values():
            patch.points_updated()
        
        if self.manager.has_bad_group_ids():
            self._handle_bad_point_groups()

        self._analyze_patches()
        
        if self.debug:
            self.debug_thing.set_frame(frame)
            self.debug_thing.patch_lookup = self.patch_lookup
            
            self.debug_thing.do_stuff(self.frame_count)

    def _analyze_patches(self):
        self.patch_analyzer.analyze(
            self.frame_count,
            self.static_patches,
            self.movable_patches)

        best_point_signals = [ ]
        
        patches = [ ]
        failed_patches = [ ]
        
        for patch in self.static_patches:
            patch.get_stats().add_frame(
                PatchFrameStats(
                    patch.get_center_point(),
                    self.frame_count,
                    patch.get_patch_seg()))
        
        for patch in self.movable_patches:
            frame_stats = PatchFrameStats(
                patch.get_center_point(),
                self.frame_count,
                patch.get_patch_segs()[0])
            frame_stats.set_stats(
                patch.get_point_count(),
                patch.get_debug_signal(),
                patch.get_score())
            
            if not patch.failed():
                patches.append(patch)
                best_point_sig = patch.get_best_point_signal()
                if best_point_sig:
                    best_point_signals.append(best_point_sig)
                frame_stats.set_state(AnalysisState.ANALYZING)
            else:
                failed_patches.append(patch)
                frame_stats.set_state(AnalysisState.RESET)
            
            patch.add_frame_stats(frame_stats)
        
        patches.sort(
            key=functools.cmp_to_key(
                movable_patch.compare_based_on_signal_score),
                reverse=True)
        
        # Take the lower quarter or half patches and treat them like failed patches
        fraction_patch_len = len(patches) // 4
        if fraction_patch_len > 0:
            #print('>>>>> Reassigning %s patches, patches: %s, failed: %s' % (fraction_patch_len, len(patches), len(failed_patches)))
            for i in range(len(patches) - fraction_patch_len, len(patches)):
                failed_patches.append(patches[i])
            patches = patches[:len(patches) - fraction_patch_len]
        
        # Re-assign failed patches
        for point_signal, failed_patch in zip(best_point_signals, failed_patches):
            if point_signal.point:
                failed_patch.reset_center_point(point_signal.point)
            else:
                failed_patch.reset_center_point(self._gen_point())
            failed_patch.frame_stats_set_state(AnalysisState.REASSIGNED)
        
        # Reset remaining failed patches
        for i in range(len(best_point_signals), len(failed_patches)):
            failed_patch = failed_patches[i]
            failed_patch.reset_center_point(self._gen_point())
            failed_patch.frame_stats_set_state(AnalysisState.RESET)
        
        # TODO Was set to 10, for debugging, using more
        top_id_count = 10 #len(patches)
        
        top_ids = [ 0 ] * top_id_count
        for i in range(min(top_id_count, len(patches))):
            top_ids[i] = patches[i].get_id()
        self.stats.add_top_patch_ids(top_ids, self.frame_count)
        
        if self.debug_prints:
            for i in range(min(10, len(patches))):
                patch = patches[i]
                sig = patch.get_debug_signal()
                print('- best patch[%2d]: ' % i, patch.get_id(), patch.get_score(), sig.bpm_est if sig else '')
        
        if self.debug:
            self.debug_thing.sorted_patches_with_signal = patches
            self.debug_thing.static_patches = self.static_patches

    def _handle_bad_point_groups(self):
        group_ids = self.manager.get_bad_group_ids()
        if self.debug_prints:
            print('reset groups: %s' % group_ids)
        for group_id in group_ids:
            # Patch IDs 0 to max static points represent the static patches,
            # implicitly. Points after that represent the movable patches.
            if group_id >= self.max_static_points:
                point = self._gen_points(1)[0]
                self.patch_lookup[group_id].reset_center_point(point)
            else:
                # Static patch, just reset it
                self.patch_lookup[group_id].reset()

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        #kernel_size = 13
        #alpha = .20 #.1 #1.0 # Simple contrast control
        #beta = 25 #50 # Simple brightness control
        #frame = cv.equalizeHist(frame)
        #frame = cv.medianBlur(frame, kernel_size)
        #frame = cv.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        #frame = cv.medianBlur(frame, kernel_size)
        #frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return frame

    def _setup(self):
        self.monitor = PointMonitor(self.info.width, self.info.height)
        self.manager = PointGroupManager(self.max_points)
        self.flow = OpticalFlow()
        self.signal_analyzer = SignalAnalyzer(self.info.fps, self.decimation)
        self.patch_analyzer = PatchAnalyzer(self.signal_analyzer)
        self.point_gen = self._create_point_gen()
        self.stats = Stats(self.info, self.decimation)
        
        self.frame_count = 0
        
        self.max_static_points = 4
        self.max_patch_points = self.max_points - self.max_static_points
        
        # Key:      patch id
        # Value:    A patch like object
        self.patch_lookup = { }
        self.static_patches = [ ]
        self.movable_patches = [ ]
        self.sorted_movable_patches = [ ]
        
        self.patch_id_gen = 0
        
        self._setup_static_points()
        
        if self.manager.get_size() != self.max_static_points:
            raise Exception('Added more static points then expected, fix it')
        
        self._setup_movable_patches()

        if self.debug:
            print('Static patch count:  %s' % len(self.static_patches))
            print('Movable patch count: %s' % len(self.movable_patches))
        
        if self.debug:
            self._setup_debug()
    
    def _setup_static_points(self):
        '''
        Add static points to the corners of the frame
        
          |-- a --|
        _ +--------------.  The frame with width and height
        | |
        b |
        | |
        - |       o <- static point
          .
          width_offset = a, height_offset = b
        '''
        static_point_width_percent = .15
        static_point_height_percent = .15
        
        width_offset = int(self.info.width * static_point_width_percent)
        height_offset = int(self.info.height * static_point_height_percent)
        
        # Top left
        p0 = Point(width_offset,                   height_offset)
        # Top right
        p1 = Point(self.info.width - width_offset, height_offset)
        # Bottom left
        p2 = Point(width_offset,                   self.info.height - height_offset)
        # Bottom right
        p3 = Point(self.info.width - width_offset, self.info.height - height_offset)
        
        max_fft_size = self.signal_analyzer.get_max_fft_size()
        avg_kernel_size = self.signal_analyzer.get_avg_kernel_size()
        num_overlaps = self.info.fps // self.decimation
        
        self._add_static_patch(
            StaticPatch(self._gen_patch_id(), p0, max_fft_size, avg_kernel_size, num_overlaps, self.manager))
        self._add_static_patch(
            StaticPatch(self._gen_patch_id(), p1, max_fft_size, avg_kernel_size, num_overlaps, self.manager))
        self._add_static_patch(
            StaticPatch(self._gen_patch_id(), p2, max_fft_size, avg_kernel_size, num_overlaps, self.manager))
        self._add_static_patch(
            StaticPatch(self._gen_patch_id(), p3, max_fft_size, avg_kernel_size, num_overlaps, self.manager))

    def _setup_movable_patches(self):
        max_patch_count = self.max_patch_points // CirclePatch.NUM_POINTS
        
        points = self._gen_points(max_patch_count)
        
        debug_one_off = False
        if debug_one_off:
            # PXL_20230824_035440033
            # Area of interest: 1003, 544
            x = 1003
            y = 544
            
            # PXL_20230825_040038487
            # Area of interest: 583, 775
            x = 583
            y = 775
            
            dist = 300
            min_x = x - dist
            max_x = x + dist
            min_y = y - dist
            max_y = y + dist
            
            gen = PrngPointGen(min_x, max_x, min_y, max_y)
            points = gen.generate(max_patch_count)
        
        if self.enable_debug_points:
            if len(points) > 100:
                points = points[:100]
            
            x0 = 50
            y0 = 700
            
            for i, point in enumerate(points):
                point.x = x0 + (i * 10)
                point.y = y0
        
        # Movable patches have a shape where one point is at the center. The
        # distance here, in pixels, is the distance from other points in the
        # movable patch to the center patch. This value was arbitrarily chosen.
        point_dist = 51
        
        max_fft_size = self.signal_analyzer.get_max_fft_size()
        avg_kernel_size = self.signal_analyzer.get_avg_kernel_size()
        num_overlaps = self.info.fps // self.decimation
        
        for point in points:
            patch = CirclePatch(
                self._gen_patch_id(),
                point,
                max_fft_size,
                avg_kernel_size,
                num_overlaps,
                self.manager,
                point_dist,
                self.signal_analyzer.get_signal_info())
            
            if self.enable_debug_points:
                patch.do_not_use_new_point_on_reset()
            
            self._add_movable_patch(patch)

    def _add_static_patch(self, patch):
        self.static_patches.append(patch)
        self._add_patch(patch)

    def _add_movable_patch(self, patch):
        self.movable_patches.append(patch)
        self.sorted_movable_patches.append(patch)
        self._add_patch(patch)

    def _add_patch(self, patch):
        self.stats.add_patch_stats(patch.get_stats())
        self.patch_lookup[patch.get_id()] = patch

    def _create_point_gen(self) -> PrngPointGen:
        width_percent = .15
        height_percent = .15
        
        min_width = int(self.info.width * width_percent)
        min_height = int(self.info.height * height_percent)
        
        max_width = self.info.width - min_width
        max_height = self.info.height - min_height
        
        return PrngPointGen(min_width, max_width, min_height, max_height)

    def _gen_points(self, num_points: int) -> list[Point]:
        return self.point_gen.generate(num_points)

    def _gen_point(self) -> Point:
        return self.point_gen.generate_point()

    def _setup_debug(self):
        from pet_breathy.debug_things import DebugThing
        self.debug_thing = DebugThing(self.info, self.prev_frame)
        
        if self.patch_analyzer.debug:
            self.patch_analyzer.debug_collector.video_info = self.info

    def _gen_patch_id(self) -> int:
        patch_id = self.patch_id_gen
        self.patch_id_gen += 1
        return patch_id
