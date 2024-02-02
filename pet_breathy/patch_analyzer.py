from pet_breathy.patch import Patch
from pet_breathy.patch_pow_2_info import PatchPow2Info
from pet_breathy import patch_pow_2_info
from pet_breathy.static_patch import StaticPatch
from pet_breathy.movable_patch import MovablePatch
from pet_breathy.signal_analyzer import SignalAnalyzer
from pet_breathy.signal_analyzer_result import SignalAnalyzerResult
from pet_breathy import signal_analyzer
from pet_breathy.signal import Signal
from pet_breathy.patch_segment import PatchSegment

from pet_breathy.debug_things import DebugPatchCollector

class PatchAnalyzer:
    def __init__(self, signal_analyzer: SignalAnalyzer):
        self.signal_analyzer = signal_analyzer
        
        self.debug = False
        self.debug_prints = False
        if self.debug:
            self.debug_collector = DebugPatchCollector()
            self.debug_collector.decimation = self.signal_analyzer.decimation

    def analyze(self,
            frame_num: int,
            static_patches: list[StaticPatch],
            movable_patches: list[MovablePatch]):
        ''' Before analysis can begin, the static patches used to analyze
        against the movable patches must have accumulated enough samples.
        There's a minimum power of 2 number of samples that must be collected.
        
        This limitation is due to the minimum number of samples to provide to
        FFT for frequency domain analysis.
        '''
        if self.debug:
            print('---')

        # sp = static patch
        sp_pow_2_infos = [ ]
        for idx, static_patch in enumerate(static_patches):
            sp_pow_2 = static_patch.get_largest_pow_of_2_point_count()
            if sp_pow_2 < self.signal_analyzer.get_min_fft_size():
                continue
            sp_pow_2_infos.append(PatchPow2Info(idx, sp_pow_2))
        
        if self.debug:
            print('static patch counts: %s' % ', '.join(str(x) for x in sp_pow_2_infos))
        
        if not sp_pow_2_infos:
            return
        
        # TODO Toggle with debug flag
        if self.debug_prints:
            print('**********************************************')
        
        if self.debug:
            self.debug_collector.add_patches(
                frame_num,
                static_patches,
                movable_patches)
        
        patch_pow_2_info.sort_infos(sp_pow_2_infos)
        
        for movable_patch in movable_patches:
            self._analyze_movable_patch(
                movable_patch,
                static_patches,
                sp_pow_2_infos)

    def done(self):
        if self.debug:
            self.debug_collector.output_to_file('./debug/data/patch_data.json')

    def set_debug_prints(self, enabled: bool):
        self.debug_prints = enabled

    def _analyze_movable_patch(self,
            movable_patch: MovablePatch,
            static_patches: list[StaticPatch],
            sp_pow_2_infos: list[PatchPow2Info]):
        
        movable_largest_pow_2 = \
            movable_patch.get_largest_pow_of_2_point_count()
        if movable_largest_pow_2 < self.signal_analyzer.get_min_fft_size():
            return
        
        #print('movable_largest_pow_2', movable_largest_pow_2)
        
        best_sp_idxs = patch_pow_2_info.find_best_infos(
            movable_largest_pow_2,
            sp_pow_2_infos)
        
        #print('best_sp_idxs', best_sp_idxs)
        
        if not best_sp_idxs:
            return
        
        # Find largest common power of 2 number of samples that at least one
        # static patch and the movable patch has accumulated
        common_pow_2 = min(
            movable_largest_pow_2,
            static_patches[best_sp_idxs[0]].get_largest_pow_of_2_point_count())
        
        results = [ ]
        
        for movable_patch_seg in movable_patch.get_patch_segs():
            result = self._analyze_movable_seg(
                movable_patch_seg,
                static_patches,
                best_sp_idxs,
                common_pow_2)
            
            results.append(result)
        
        movable_patch.analyze_new_signals(results)

    def _analyze_movable_seg(self,
            movable_seg: PatchSegment,
            static_patches: list[StaticPatch],
            best_sp_idxs: list[int],
            common_pow_2: int) -> SignalAnalyzerResult:
        
        dy_coords = [ ]
        
        for best_sp_idx in best_sp_idxs:
            static_seg = static_patches[best_sp_idx].get_patch_seg()
            
            static_avg_y_coord = \
                static_seg.get_segment().get_latest_avg_y_coords(common_pow_2)
            static_act_y_coord = \
                static_seg.get_segment().get_latest_act_y_coords(common_pow_2)
            
            movable_avg_y_coord = \
                movable_seg.get_segment().get_latest_avg_y_coords(common_pow_2)
            movable_act_y_coord = \
                movable_seg.get_segment().get_latest_act_y_coords(common_pow_2)
            
            # The following mitigates the camera as it moves. The idea is that
            # if you take static patch dx/dy from movabable patch, you would
            # only be left with the movement of the subject you're interested
            # in. It doesn't quite work that way. Instead what I see is that
            # all the points tend to drift in certain directions as the camera
            # moves. Not all patches move at the same rate, same distance etc.
            avg_diff_y_coord = static_avg_y_coord - movable_avg_y_coord
            dy_coord = (static_act_y_coord - movable_act_y_coord) - avg_diff_y_coord
            
            dy_coords.append(dy_coord)
        
        return self.signal_analyzer.calc_avg_signals(
            dy_coords,
            movable_seg.get_overlapper())
