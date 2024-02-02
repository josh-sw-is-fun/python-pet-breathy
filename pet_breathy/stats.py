from pet_breathy.video_info import VideoInfo
from pet_breathy.patch_stats import PatchStats

from typing import List
import json


class Stats:
    def __init__(self, info: VideoInfo, decimation: int):
        self.info = info
        self.decimation = decimation
        
        # Key:      patch id
        # Value:    PatchStat
        self.patch_stats_lookup = { }
        
        # List of list of patch ids where the first patch id in the list has the top score
        # Key:      Frame number
        # Value:    Top patch ids, list
        self.top_patch_ids = { }

    def add_patch_stats(self, patch_stats: PatchStats):
        if patch_stats.patch_id in self.patch_stats_lookup:
            raise RuntimeError('Cannot add patch stats, already added')
        self.patch_stats_lookup[patch_stats.patch_id] = patch_stats

    def add_top_patch_ids(self, top_ids: List[int], frame_num: int):
        self.top_patch_ids[frame_num] = top_ids


def write_to_file(
        stats: Stats,
        file_name: str,
        video_file_name: str,
        start_frame_num: int,
        stop_frame_num: int):
    patch_stats_lookup = { }
    
    for patch_id, patch_stat in stats.patch_stats_lookup.items():
        frames = { }
        for frame in patch_stat.frames:
            sig_dict = { }
            if frame.sig is not None:
                sig_dict = {
                    'strength': str(frame.sig.strength),
                    'bpm_est': str(frame.sig.bpm_est),
                    'bpm_precision': str(frame.sig.bpm_precision),
                    'fft_size': frame.sig.fft_size,
                    'fft_level': frame.sig.fft_level,
                    'decimation': frame.sig.decimation,
                }
            sig_score_dict = { }
            if frame.sig_score is not None:
                sig_score_dict = {
                    'score': frame.sig_score.score,
                    'fft_level': frame.sig_score.fft_level,
                    'strength': str(frame.sig_score.sig_strength),
                }
            if frame.extended_info:
                y_avg_pts = list(map(str, frame.y_avg_pts))
                y_act_pts = list(map(str, frame.y_act_pts))
                if frame.spectra is not None:
                    spectra = list(map(str, frame.spectra))
                else:
                    spectra = [ ]
            else:
                y_avg_pts = [ ]
                y_act_pts = [ ]
                spectra = [ ]
            frames[frame.frame_num] = {
                'point': [ str(frame.point.x), str(frame.point.y) ],
                'state': str(frame.state),
                'seg_length': frame.seg_length,
                'signal': sig_dict,
                'signal_score': sig_score_dict,
                'y_avg_pts': y_avg_pts,
                'y_act_pts': y_act_pts,
                'spectra': spectra,
            }
        patch_stats_lookup[f'{patch_id}'] = {
            'patch_id': patch_stat.patch_id,
            'patch_type': str(patch_stat.patch_type),
            'frames': frames
        }
    
    stats_dict = {
        'info': {
            'fps': stats.info.fps,
            'width': stats.info.width,
            'height': stats.info.height,
            'frame_count': stats.info.frame_count,
            'video_path': video_file_name,
            'start_frame_num': start_frame_num,
            'stop_frame_num': stop_frame_num,
        },
        'decimation': stats.decimation,
        'patch_stats_lookup': patch_stats_lookup,
        'top_patch_ids': stats.top_patch_ids,
    }
    
    with open(file_name, 'w') as fout:
        json.dump(stats_dict, fout)
