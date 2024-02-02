from pet_breathy.video_display import VideoDisplay
from pet_breathy.video_reader import VideoReader
from pet_breathy import video_reader
from pet_breathy.pet_breathy import PetBreathy
from pet_breathy import optical_flow_point_budget_calculator

import cv2 as cv
import numpy as np
import datetime as dt
import enum

class ExitStatus(enum.Enum):
    DoNothing = enum.auto()
    JustExit = enum.auto()
    PauseAndExit = enum.auto()

class PetBreathyRunner:
    def __init__(self, reader: VideoReader):
        self.reader = reader
        self.info = self.reader.get_info()
        
        if self.info.fps != 30:
            # TODO If 60 fps, could just decimate by 2x ... or take advantage
            # of 60 fps!
            raise Exception('Expecting FPS to be 30, '
                'instead it is: %s' % self.info.fps)
        
        self.display = VideoDisplay('main', self.info.width, self.info.height, 0.5)

    @staticmethod
    def create_from_video_file(file_path: str):
        reader = video_reader.create_video_file_reader(file_path)
        return PetBreathyRunner(reader)

    def run(self):
        self.running = True
        try:
            self._run()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
    
    def _run(self):
        f0 = self.reader.get_next_frame()
        f1 = self.reader.get_next_frame()
        
        # Need to get the max points from here
        max_points, decimation = self._update_point_budget(f0, f1)
        
        gen_constant_max_points = True
        if gen_constant_max_points:
            max_points = 796
        print('Max points: %s' % max_points)
        
        debug = True
        self.breathy = PetBreathy(self.info, max_points, decimation, f1, debug)
        
        max_runtime = self.breathy.get_max_runtime_per_frame()
        
        self.start_time = dt.datetime.now()
        
        pause_before_exiting = False
        
        while self.running:
            t0 = dt.datetime.now()
            
            self._process_frame()
            
            t1 = dt.datetime.now()
            
            frame_time = (t1 - t0).total_seconds()
            if frame_time > max_runtime:
                print('Took too long! %.3f sec' % frame_time)
            
            exit_status = self._wait_key_press(frame_time)
            
            if exit_status != ExitStatus.DoNothing:
                if exit_status == ExitStatus.PauseAndExit:
                    pause_before_exiting = True
                break
        
        if pause_before_exiting:
            while self._wait_key_press_with_timeout(10) == ExitStatus.DoNothing:
                pass
        
        self.breathy.done()
    
    def _process_frame(self):
        frame = self.reader.get_next_frame()
        
        if frame is not None:
            frame = self.breathy.process_frame(frame)
            
            self.display.show(frame)
        else:
            self.running = False
    
    def _update_point_budget(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> int:
        point_budget_calc = \
            optical_flow_point_budget_calculator.create(prev_frame, curr_frame)
        
        decimation = 6
        max_points = .35 * point_budget_calc.calc_points(decimation / self.info.fps)
        
        if max_points < 0:
            raise RuntimeError('Max points went negative, 1st measurement was '
                'greater than 2nd measurement? Max points: %s' % max_points)
        
        return int(max_points), decimation
    
    def _wait_key_press(self, frame_elapsed_time: float) -> bool:
        frame_rate = 1.0 / self.info.fps
        
        actual_elapsed_time = (dt.datetime.now() - self.start_time).total_seconds()
        expected_elapsed_time = frame_rate * self.reader.get_frame_count()
        
        off_time = expected_elapsed_time - actual_elapsed_time
        
        wait_time_ms = max(1, int(off_time * 1000))
        
        return self._wait_key_press_with_timeout(wait_time_ms)
    
    def _wait_key_press_with_timeout(self, wait_time_ms):
        k = cv.waitKey(wait_time_ms)
        if k == 99: # 'c'
            exit_status = ExitStatus.JustExit
        elif k == 115: # 's'
            exit_status = ExitStatus.PauseAndExit
        else:
            exit_status = ExitStatus.DoNothing
        return exit_status


