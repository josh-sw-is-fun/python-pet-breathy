from pet_breathy.video_file_reader import VideoFileReader
from pet_breathy.video_info import VideoInfo

import numpy as np

class VideoReader:
    # TODO Fix annotation of reader, an obj that supports get_info, get_next_frame
    def __init__(self, reader):
        self.reader = reader
        self.frame_num = 0

    def get_info(self) -> VideoInfo:
        return self.reader.get_info()

    def get_next_frame(self) -> np.ndarray:
        frame = self.reader.get_next_frame()
        if frame is not None:
            self.frame_num += 1
        return frame

    def get_frame_count(self) -> int:
        return self.frame_num

def create_video_file_reader(video_file_path: str) -> VideoReader:
    reader = VideoFileReader(video_file_path)
    return VideoReader(reader)
