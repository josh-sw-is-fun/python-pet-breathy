import numpy as np
import scipy as sp

class FftOverlapper:
    def __init__(self, num_overlaps):
        self.num_overlaps = num_overlaps
        self.window = None
        
        self.debug = False
        self.last_raw_fft = None
        self.sum_spectra = 0
        self.spectras_count = 0

    def add_vals(self, vals):
        vals_len = len(vals)
        if self.window is None or self.window.size != vals_len:
            self.window = self._create_window(vals_len)
            self._reset_spectra()
        
        if self.debug:
            self.last_raw_fft = self._calc_fft(vals)
        self.last_fft = self._calc_fft(np.multiply(vals, self.window))
        spectra = (self.last_fft / vals_len * 2.) ** 2.
        
        try:
            self.spectras[self.spectras_idx] = spectra
        except IndexError:
            print(f'len: {len(self.spectras)}, {self.spectras_idx=}')
            raise
        self.spectras_idx = (self.spectras_idx + 1) % self.num_overlaps
        self.spectras_count += 1
        
        if self.spectras_count > self.num_overlaps:
            self.sum_spectra -= self.spectras[self.spectras_idx]
        self.sum_spectra += spectra

    def get_spectra(self):
        count = min(self.spectras_count, self.num_overlaps)
        if count:
            return np.sqrt(self.sum_spectra / count)
        return None

    def get_last_fft(self):
        return self.last_fft

    def get_last_raw_fft(self):
        return self.last_raw_fft

    def get_spectras_count(self):
        return self.spectras_count

    def _create_window(self, fft_len):
        return np.hanning(fft_len)

    def _reset_spectra(self):
        self.spectras = [ 0 ] * self.num_overlaps
        self.spectras_count = 0
        self.spectras_idx = 0
        self.sum_spectra = 0

    def _calc_fft(self, y: np.ndarray) -> np.ndarray:
        yf = sp.fft.fft(y)
        yf = 2.0 / len(y) * np.abs(yf[:len(y) // 2])
        return yf

'''
    def _calc_spectra(self, start_step, end_step):
        spectra = 0
        spectra_count = 0.0
        #for i in range(self.fft_len // 2, len(self.dy_coord_avgs), self.fft_len // 2):
        #    if i + self.fft_len // 2 > len(self.dy_coord_avgs):
        #        break
        print('start_step:', start_step, 'end_step:', end_step)
        for i in range(start_step, end_step):
            print('>>>> len(self.dy_coord_avgs[i]):', len(self.dy_coord_avgs[i]), 'i:', i)
            curr_a = np.multiply(self.dy_coord_avgs[i], self.fft_win)
            curr_b = _calc_fft(curr_a) / self.fft_len * 2.
            curr_c = (curr_b) ** 2.
            spectra = curr_c + spectra
            spectra_count += 1.0
        spectra = np.sqrt(spectra / spectra_count)
        return spectra
'''

