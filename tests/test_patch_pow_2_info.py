from pet_breathy.patch_pow_2_info import PatchPow2Info
from pet_breathy import patch_pow_2_info

import unittest
import numpy as np

class TestPatchPow2Info(unittest.TestCase):
    def test_sort(self):
        infos = [
            PatchPow2Info(3, 128),
            PatchPow2Info(2, 256),
            PatchPow2Info(0, 64),
            PatchPow2Info(1, 1024),
            PatchPow2Info(4, 64),
        ]
        
        patch_pow_2_info.sort_infos(infos)
        
        self.assertEqual(1, infos[0].idx)
        self.assertEqual(1024, infos[0].pow_2)
        
        self.assertEqual(2, infos[1].idx)
        self.assertEqual(256, infos[1].pow_2)
        
        self.assertEqual(3, infos[2].idx)
        self.assertEqual(128, infos[2].pow_2)
        
        self.assertTrue(infos[3].idx == 0 or infos[3].idx == 4)
        self.assertEqual(64, infos[3].pow_2)
        
        self.assertTrue(infos[3].idx == 0 or infos[3].idx == 4)
        self.assertEqual(64, infos[4].pow_2)

    def test_find_best_infos_empty(self):
        infos = [ ]
        best_idxs = patch_pow_2_info.find_best_infos(16, [ ])
        self.assertEqual(0, len(best_idxs))

    def test_find_best_infos_all_not_good(self):
        infos = [ PatchPow2Info(3, 128) ]
        best_idxs = patch_pow_2_info.find_best_infos(16, [ ])
        self.assertEqual(0, len(best_idxs))

    def test_find_best_infos_one_good(self):
        infos = [ PatchPow2Info(3, 128) ]
        best_idxs = patch_pow_2_info.find_best_infos(128, infos)
        
        self.assertEqual(1, len(best_idxs))
        self.assertEqual(3, best_idxs[0])

    def test_find_best_infos_good_among_multiple(self):
        infos = [
            PatchPow2Info(123, 128),
            PatchPow2Info(412, 128),
            PatchPow2Info(12, 64),
            ]
        best_idxs = patch_pow_2_info.find_best_infos(128, infos)
        
        self.assertEqual(2, len(best_idxs))
        self.assertEqual(123, best_idxs[0])
        self.assertEqual(412, best_idxs[1])

    def test_find_best_infos_two_good_enough(self):
        infos = [
            PatchPow2Info(123, 128),
            PatchPow2Info(412, 64),
            PatchPow2Info(242, 32),
            ]
        best_idxs = patch_pow_2_info.find_best_infos(64, infos)
        
        self.assertEqual(2, len(best_idxs))
        self.assertEqual(123, best_idxs[0])
        self.assertEqual(412, best_idxs[1])

    def test_find_best_infos_all_good_and_larger(self):
        infos = [
            PatchPow2Info(123, 128),
            PatchPow2Info(412, 64),
            ]
        best_idxs = patch_pow_2_info.find_best_infos(32, infos)
        
        self.assertEqual(2, len(best_idxs))
        self.assertEqual(123, best_idxs[0])
        self.assertEqual(412, best_idxs[1])

    def test_find_best_infos_best_candidates(self):
        ''' Even though we're looking for 128, the best two consecutive infos are 64 '''
        infos = [
            PatchPow2Info(123, 64),
            PatchPow2Info(412, 64),
            ]
        best_idxs = patch_pow_2_info.find_best_infos(128, infos)
        
        self.assertEqual(2, len(best_idxs))
        self.assertEqual(123, best_idxs[0])
        self.assertEqual(412, best_idxs[1])

    def test_find_best_infos_best_consecutive_candidates(self):
        ''' Even though we're looking for 128, the best two consecutive infos
        are 64, last one is different so it shouldn't be included
        '''
        infos = [
            PatchPow2Info(123, 64),
            PatchPow2Info(412, 64),
            PatchPow2Info(200, 32),
            ]
        best_idxs = patch_pow_2_info.find_best_infos(128, infos)
        
        self.assertEqual(2, len(best_idxs))
        self.assertEqual(123, best_idxs[0])
        self.assertEqual(412, best_idxs[1])



