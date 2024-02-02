from pet_breathy.video_info import VideoInfo

import cv2 as cv
import numpy as np

class VideoFileReader:
    def __init__(self, file_path: str):
        cap = cv.VideoCapture(cv.samples.findFileOrKeep(file_path))
        
        if not cap.isOpened():
            raise Exception('Could not open file: %s' % file_path)
        
        # Bug in OpenCV introduced when I upgraded to 4.11, I do not remember what the prev version of OpenCV was that worked
        # I shouldn't have to specify this. The bug is that they changed this behavior.
        cap.set(cv.CAP_PROP_ORIENTATION_AUTO, 1.0)
        
        self.file_path = file_path
        self.cap = cap
        
        self.info = VideoInfo()
        self.info.fps = int(round(cap.get(cv.CAP_PROP_FPS)))
        self.info.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.info.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.info.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    def get_next_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return frame
        return None
    
    def get_info(self) -> VideoInfo:
        return self.info
