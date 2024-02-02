'''
PYTHONPATH=$(pwd) python3 ./tests/regression/pet_breathy_regression_tester.py --meta-data-path ./tests/regression/meta_data.json --relative-video-dir ../videos
'''
from pet_breathy.video_info import VideoInfo
from pet_breathy.video_file_reader import VideoFileReader
from pet_breathy.video_reader import VideoReader
from pet_breathy.pet_breathy import PetBreathy
from pet_breathy.patch_type import PatchType
from pet_breathy import stats

import sys
import argparse
import os
from dataclasses import dataclass
import json
import numpy as np

def _main():
    args = _parse_args()
    run(
        args.meta_data_path,
        args.relative_video_dir,
        args.out_stats_dir)

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Runs regression tests for pet-breathy')

    parser.add_argument(
        '--meta-data-path',
        help='File path to meta data')

    parser.add_argument(
        '--relative-video-dir',
        help='Meta data file has relative video file paths, the file paths are relative ot this dir')

    parser.add_argument(
        '--out-stats-dir',
        help='Path of stats .json files will be saved')

    return parser.parse_args()

@dataclass
class VideoMetaData:
    file_name: str
    expected_bpm: float
    breath_count: int
    examine_begin_sec: int
    examine_end_sec: int
    notes: str

def run(meta_data_path, relative_video_dir, out_stats_dir):
    if not relative_video_dir:
        relative_video_dir = ''

    with open(meta_data_path) as fin:
        data = json.load(fin)

    final_result = True
    fail_test_count = 0
    test_count = len(data['videos'])

    for video_data in data['videos']:
        meta_data = VideoMetaData(**video_data)
        meta_data.file_name = os.path.join(relative_video_dir, meta_data.file_name)
        
        #if not '20231003_162545_sophie' in meta_data.file_name:
        #    continue
        
        result = run_test(meta_data, out_stats_dir)
        
        if result:
            print('-- PASS --')
        else:
            print('-- FAIL --')
            final_result = False
            fail_test_count += 1
        
        #break
    
    if final_result:
        print('All tests passed')
    else:
        print('Failed tests: %s of %s' % (fail_test_count, test_count))

def run_test(meta_data: VideoMetaData, out_json_dir):
    reader = SnippetVideoFileReader(
        meta_data.file_name,
        meta_data.examine_begin_sec,
        meta_data.examine_end_sec)
    print('-------------------------------------------------------------------------------')
    print(f'- File name: {meta_data.file_name}, '
        f'Exp bpm: {meta_data.expected_bpm}, '
        f'[{meta_data.examine_begin_sec} {meta_data.examine_end_sec}] [Begin End]')
    print(f'             {reader.get_info()}')
    
    vid_reader = VideoReader(reader)
    
    # Could make a special runner that doesn't have the gui elements
    # Or refactor PetBreathyRunner
    runner = PetBreathyRunnerTester(vid_reader, meta_data)
    result = runner.run()
    
    start_frame, stop_frame = reader.get_start_stop_frame_num()
    
    if out_json_dir is not None:
        file_name = os.path.basename(meta_data.file_name)
        name, _ = os.path.splitext(file_name)
        stats.write_to_file(
            runner.get_stats(),
            os.path.join(out_json_dir, f'{name}.json'),
            meta_data.file_name,
            start_frame,
            stop_frame)
    
    return result

class PetBreathyRunnerTester:
    def __init__(self, vid_reader, meta_data):
        self.reader = vid_reader
        self.meta_data = meta_data
        
        decimation = 6
        max_points = 500
        prev_frame = self.reader.get_next_frame()
        
        self.debug_prints = False

        self.breathy = PetBreathy(self.reader.get_info(), max_points, decimation, prev_frame)
        self.breathy.set_debug_prints(self.debug_prints)

    def run(self):
        self.running = True
        
        while self.running:
            self._process_frame()
        
        breathy_stats = self.breathy.get_stats()
        
        if self.debug_prints:
            print('Info: %s' % breathy_stats.info)
            print('Decimation: %s' % breathy_stats.decimation)

            top_ids = breathy_stats.top_patch_ids[max(breathy_stats.top_patch_ids.keys())]
            for top_id in top_ids:
                patch_stats = breathy_stats.patch_stats_lookup[top_id]
                last_frame = patch_stats.frames[-1]
                print('- ID: %s, state: %s' % (patch_stats.patch_id, last_frame.state))
                print('  - Signal: %s' % last_frame.sig)
                print('  - Score:  %s' % last_frame.sig_score)

        # top_ids will be sorted in descending order
        top_ids = breathy_stats.top_patch_ids[max(breathy_stats.top_patch_ids.keys())]
        patch_stats = breathy_stats.patch_stats_lookup[top_ids[0]]
        last_frame = patch_stats.frames[-1]
        
        exp_min_bpm = self.meta_data.expected_bpm - 2
        exp_max_bpm = self.meta_data.expected_bpm + 2
        if exp_min_bpm <= last_frame.sig.bpm_est <= exp_max_bpm:
            result = True
        else:
            result = False
            
            for top_id in top_ids:
                patch_stats = breathy_stats.patch_stats_lookup[top_id]
                last_frame = patch_stats.frames[-1]
                print('- ID: %s, state: %s' % (patch_stats.patch_id, last_frame.state))
                print('  - Signal: %s' % last_frame.sig)
                print('  - Score:  %s' % last_frame.sig_score)
        
        return result
    
    def get_stats(self):
        return self.breathy.get_stats()
    
    def _process_frame(self):
        frame = self.reader.get_next_frame()
        
        if frame is not None:
            self.breathy.process_frame(frame)
        else:
            self.running = False

class SnippetVideoFileReader:
    def __init__(self, file_path, start_sec, stop_sec):
        self.reader = VideoFileReader(file_path)
        self.frame_num = 0
        
        self.info = self.reader.get_info().clone()
        
        #print(f'  -> actual: {self.reader.get_info()}')
        
        self.start_frame = int(start_sec * self.info.fps) if start_sec != -1 else 0
        self.stop_frame = int(stop_sec * self.info.fps) if stop_sec != -1 else self.info.frame_count
        
        self.info.frame_count = self.stop_frame - self.start_frame
        
        if self.start_frame > 0:
            # Eat frames until we catch up:
            for i in range(self.start_frame - 1):
                self._get_next_frame()

    def get_next_frame(self) -> np.ndarray:
        if self.frame_num < self.info.frame_count:
            return self._get_next_frame()
        return None

    def get_info(self) -> VideoInfo:
        return self.info

    def get_start_stop_frame_num(self) -> tuple[int, int]:
        return self.start_frame, self.stop_frame

    def _get_next_frame(self) -> np.ndarray:
        frame = self.reader.get_next_frame()
        if frame is not None:
            self.frame_num += 1
        return frame

if __name__ == '__main__':
    _main()
