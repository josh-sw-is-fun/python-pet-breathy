from pet_breathy.signal import Signal

import numpy as np

class SignalAnalyzerResult:
    def __init__(self, signals: list[Signal], yf_points: np.ndarray):
        self.signals = signals
        self.yf_points = yf_points
