import sys
import numpy as np
from matplotlib import pyplot as plt
import scipy

from pet_breathy.signal_analyzer import SignalAnalyzer

def main():
    args = sys.argv[1:]
    _run(*args)

def _run(*args):
    analyzer = AnalyzeFftData(*args)
    analyzer.run()

class AnalyzeFftData:
    def __init__(self, in_data_path):
        self.in_data_path = in_data_path
        
        self.max_num_seg_groups = 500
        
        fps = 30
        decimation = 6
        self.analyzer = SignalAnalyzer(fps, decimation)
    
    def run(self):
        with open(self.in_data_path) as self.fh:
            self._run()
    
    def _run(self):
        self._parse_seg_groups()
        
        print('Number of segment groups: %s' % self.seg_groups.get_num_groups())
        
        #self._do_avg()
        self._examine_each_group()
        #self._avg_segs_then_fft()
    
    def _avg_segs_then_fft(self):
        for seg_group in self.seg_groups.get_groups():
            segs = seg_group.get_segs()
            avg_segs = segs[0]
            for i in range(1, len(segs)):
                avg_segs += segs[i]
            avg_segs /= len(segs)
            
            fig, axs = plt.subplots(nrows=1, ncols=2)
            
            yf = self.analyzer.calc_avg_fft([avg_segs])
            axs[0].plot(
                np.arange(len(avg_segs)),
                avg_segs,
                alpha=.30
                )
            axs[1].plot(
                np.arange(len(yf)),
                yf,
                alpha=.30
                )
            
            plt.show()
    
    def _examine_each_group(self):
        for seg_group in self.seg_groups.get_groups():
            segs = seg_group.get_segs()
            
            fig, axs = plt.subplots(nrows=2, ncols=max(2, len(segs)))
            
            print('Num segs: %s' % len(segs))
            
            for seg_idx, seg in enumerate(segs):
                axs[0][0].plot(
                    np.arange(len(seg)),
                    seg,
                    alpha=.30
                    )
                
                yf = self.analyzer.calc_avg_fft([seg])
                axs[1][seg_idx].plot(
                    np.arange(len(yf)),
                    yf)
            
            yf = self.analyzer.calc_avg_fft(
                segs
                #[segs[2]]
                )
            axs[0][1].plot(
                np.arange(len(yf)),
                yf)
            
            #axs[0].set_ylim([0, 2])
            plt.show()
    
    def _do_avg(self):
        yf_count = 0
        yf_avg = None
        for seg_group in self.seg_groups.get_groups():
            segs = seg_group.get_segs()
            
            #print('num segs: %s' % len(segs))
            
            yf = self.analyzer.calc_avg_fft(segs)
            yf_count += 1
            if yf_avg is not None:
                yf_avg += yf
            else:
                yf_avg = yf
            
            if False:
                #x = np.arange(len(yf_avg))
                plt.plot(
                    #self.analyzer.get_xfft(64),
                    np.arange(len(yf_avg)),
                    yf_avg / yf_count,
                #    #alpha=.30
                    )
                plt.show()
        
        yf_avg /= yf_count
        #x = np.arange(len(yf_avg))
        plt.plot(
            #self.analyzer.get_xfft(64),
            np.arange(len(yf_avg)),
            yf_avg,
            #alpha=.30
            )
        
        plt.show()

    def _parse_seg_groups(self):
        line_num = 1
        state = 0
        self.seg_groups = SegGroups()
        seg_group = SegGroup()
        seg = [ ]
        
        self.done_parsing = False
        
        for line in self.fh:
            line_num += 1
            line = line.strip()
            
            if line:
                if '[' in line:
                    state = 1
                elif ']' in line:
                    state = 2
                else:
                    state = 3
                
                if state == 1:
                    line = line.lstrip('[')
                elif state == 2:
                    line = line.rstrip(']')
                seg += [float(v) for v in line.split()]
                
                if state == 2:
                    seg_group.add(np.asarray(seg))
                    seg = [ ]
            else:
                if seg_group:
                    self.seg_groups.add(seg_group)
                    
                    if self.seg_groups.get_num_groups() >= self.max_num_seg_groups:
                        break
                seg_group = SegGroup()

class SegGroups:
    def __init__(self):
        self.seg_groups = [ ]

    def add(self, seg_group):
        if seg_group.get_seg_len() == 16:
            if seg_group.get_num_segs() > 2:
                self.seg_groups.append(seg_group)
    
    def get_num_groups(self):
        return len(self.seg_groups)
    
    def get_groups(self):
        return self.seg_groups

class SegGroup:
    def __init__(self):
        self.segs = [ ]

    def add(self, seg):
        self.segs.append(seg)
    
    def get_seg_len(self):
        return len(self.segs[0])
    
    def get_num_segs(self):
        return len(self.segs)
    
    def get_segs(self):
        return self.segs

        '''
[ 1.45701731  2.11945969  1.37001178  0.36332169 -0.14144026  0.07391575
 -0.20980835 -0.39970616 -0.57044111 -0.24790301 -0.26672581 -0.22905186
 -0.2920336  -0.58625539 -0.81373901 -0.90502548]
[ 2.27726878  3.27979294  2.69904722  0.71085151  0.28954642 -0.35311018
 -1.57679531 -0.90477208 -0.51098633  0.13087681  0.11708723 -0.54963902
 -0.54221017 -0.78117879 -0.98690796 -2.43916321]
        '''

if __name__ == '__main__':
    main()
