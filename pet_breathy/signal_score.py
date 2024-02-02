from pet_breathy.fft_levels import FftLevel

class SignalScore:
    WORST_SCORE = -1
    def __init__(self):
        self.reset()

    def __str__(self):
        return f'score: {self.score}, FFT level: {self.fft_level}, sig strength: {self.sig_strength}'

    def got_lock(self, fft_level, sig_strength):
        if self.fft_level != fft_level:
            self.score = 0
        elif self.score < 3:
            self.score += 1
        self.fft_level = fft_level
        self.sig_strength = sig_strength

    def lost_signal_lock(self):
        if self.score > SignalScore.WORST_SCORE:
            self.score -= 1

    def is_signal_lost(self):
        return self.score == SignalScore.WORST_SCORE

    def reset(self):
        self.score = 0
        self.fft_level = FftLevel.LEVEL_1
        self.sig_strength = 0.0

def compare(s0, s1):
    '''
    TODO Could swap score and fft_level
    TODO Could multply sig_strength and score to get weighted score
         - Smaller fft_level will have a higher weighted score
         - Could go, fft_level then weighted score 
    '''
    if s0.score < s1.score:
        return -1
    if s0.score > s1.score:
        return 1
    if s0.fft_level < s1.fft_level:
        return -1
    if s0.fft_level > s1.fft_level:
        return 1
    if s0.sig_strength < s1.sig_strength:
        return -1
    if s0.sig_strength > s1.sig_strength:
        return 1
    return 0
