
class SignalInfo:
    def __init__(self):
        self.decimation = 0
        self.fps = 0

    def __str__(self):
        return 'decimation: %s, fps: %s' % (self.decimation, self.fps)

    def get_decimated_fps(self):
        if self.decimation != 0:
            return self.fps // self.decimation
        return 0
