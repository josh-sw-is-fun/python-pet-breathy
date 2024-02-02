import cv2 as cv
import numpy as np

class VideoDisplay:
    def __init__(self, name: str, width: int, height: int, ratio: float):
        self.name = name
        
        cv.namedWindow(self.name, cv.WINDOW_NORMAL)
        
        cv.resizeWindow(
            self.name,
            int(width * ratio),
            int(height * ratio))
    
    def show(self, frame: np.ndarray):
        cv.imshow(self.name, frame)

    def move(self, x: int, y: int):
        cv.moveWindow(self.name, x, y)

    def close(self):
        cv.destroyWindow(self.name)

    def is_open(self) -> bool:
        return cv.getWindowProperty(self.name,  cv.WND_PROP_VISIBLE) > 0
