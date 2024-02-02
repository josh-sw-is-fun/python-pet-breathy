import copy

class VideoInfo:
    def __init__(self):
        self.fps = 0
        self.width = 0
        self.height = 0
        self.frame_count = 0

    def __str__(self):
        return (f'width: {self.width}, height: {self.height}, '
            f'fps: {self.fps}, frames: {self.frame_count}, '
            f'runtime: {self.frame_count / self.fps} secs')

    def clone(self):
        return copy.copy(self)
