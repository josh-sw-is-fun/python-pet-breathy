from pet_breathy.point import Point
from pet_breathy.signal import Signal

class PointSignal:
    def __init__(self, point: Point, signal: Signal):
        self.point = point
        self.signal = signal
