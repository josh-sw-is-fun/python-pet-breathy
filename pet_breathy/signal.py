
class Signal:
    def __init__(self):
        self.strength = 0.0
        self.bpm_est = 0.0
        self.bpm_precision = 0.0
        self.fft_size = 0
        self.fft_level = None
        self.decimation = 0
        #self.prev_fft_level_has_energy = False
        #self.next_fft_level_has_energy = False

    def __str__(self) -> str:
        return 'strength: %s, bpm est: %s, fft size: %s, fft level: %s' % (
            self.strength, self.bpm_est, self.fft_size, self.fft_level)
