from pet_breathy.patch_cluster import PatchCluster, PatchClusterFrame
from pet_breathy.point import Point

import unittest
import numpy as np

class TestPatchCluster(unittest.TestCase):
    def test_add_2nd_frame(self):
        ''' Tests adding the same frame to the cluster twice. Should be added no prob since distance, bpm, and frame num are all the same.
        '''
        radius = 10
        bpm_threshold = 2
        bpm_strength = 1
        frame_num = 0
        cluster_id = 0

        f0 = PatchClusterFrame(Point(0, 0), 0, bpm_strength, frame_num)
        c0 = PatchCluster(cluster_id, radius, bpm_threshold, f0)
        self.assertEqual(1, c0.get_current_frame_count())
        
        self.assertTrue(c0.add_frame(f0))
        self.assertEqual(2, c0.get_current_frame_count())

    def test_2nd_patch_cannot_be_added_to_1st(self):
        ''' Tests a 2nd patch is outside radius, bpm_threshold so it cannot be added to the 1st cluster.
        '''
        radius = 10
        bpm_threshold = 2
        bpm_strength = 1
        frame_num = 0
        cluster_id = 0
        
        f0 = PatchClusterFrame(Point(0, 0), 0, bpm_strength, frame_num)
        c0 = PatchCluster(cluster_id, radius, bpm_threshold, f0)
        self.assertEqual(1, c0.get_current_frame_count())

        f1 = PatchClusterFrame(Point(0, 0), 0, bpm_strength, frame_num)
        # Point too far away to cluster
        f1.center = Point(radius * 2, 0)
        
        self.assertFalse(c0.add_frame(f1))
        self.assertEqual(1, c0.get_current_frame_count())

        f1 = PatchClusterFrame(Point(0, 0), 0, bpm_strength, frame_num)
        # BPM less than threshold
        f1.bpm = f0.bpm - (bpm_threshold + 1)
        self.assertFalse(c0.add_frame(f1))
        
        # BPM is greator than threshold
        f1.bpm = f0.bpm + (bpm_threshold + 1)
        self.assertFalse(c0.add_frame(f1))

    def test_new_frame(self):
        '''Tests adding a new frame to the existing cluster.'''
        radius = 10
        bpm_threshold = 2
        bpm_strength = 1
        frame_num = 0
        cluster_id = 0
        
        f0 = PatchClusterFrame(Point(0, 0), 0, bpm_strength, frame_num)
        c0 = PatchCluster(cluster_id, radius, bpm_threshold, f0)
        
        # New frame, but same frame number. After adding the frame, there should be 2 frames in
        # the same cluster for the same frame number
        f1 = PatchClusterFrame(Point(0, 0), 0, bpm_strength, frame_num)
        self.assertTrue(c0.add_frame(f1))
        self.assertEqual(2, c0.get_current_frame_count())
        
        # New frame, different frame number
        f2 = PatchClusterFrame(Point(0, 0), 0, bpm_strength, frame_num + 1)
        self.assertTrue(c0.add_frame(f2))
        
        # The frame count goes back down to 1
        self.assertEqual(1, c0.get_current_frame_count())

        # Checking that the frame count for frame_num didn't change
        self.assertEqual(2, c0.get_frame_count(frame_num))

    def test_center_moves_after_new_frame(self):
        '''Tests that the clusters center moves when a new frame is added but not when a 2nd frame
        is added with the same frame_num as the 1st.
        '''
        radius = 10
        bpm_threshold = 2
        bpm_strength = 1
        frame_num = 0
        cluster_id = 0
        
        f0 = PatchClusterFrame(Point(1, 2), 0, bpm_strength, frame_num)
        c0 = PatchCluster(cluster_id, radius, bpm_threshold, f0)
        
        center = c0.get_current_center()
        self.assertEqual(1, center.x)
        self.assertEqual(2, center.y)
        
        # 2nd frame with the same frame_num but different center shouldn't change the center of the cluster
        f1 = PatchClusterFrame(Point(3, 4), 0, bpm_strength, frame_num)
        self.assertTrue(c0.add_frame(f1))
        
        center = c0.get_current_center()
        self.assertEqual(1, center.x)
        self.assertEqual(2, center.y)
        
        # Add another frame but with a different center, still within the cluster, and a different
        # frame num. The center should move to the new patch
        f2 = PatchClusterFrame(Point(2, 3), 0, bpm_strength, frame_num + 1)
        self.assertTrue(c0.add_frame(f2))
        
        center = c0.get_current_center()
        self.assertEqual(2, center.x)
        self.assertEqual(3, center.y)
        
        # Add another frame with the same frame_num + 1 with different center but can still be
        # added to the same cluster. Make sure center doesn't move
        f3 = PatchClusterFrame(Point(5, 4), 0, bpm_strength, frame_num + 1)
        self.assertTrue(c0.add_frame(f3))
        
        center = c0.get_current_center()
        self.assertEqual(2, center.x)
        self.assertEqual(3, center.y)
        
        self.assertEqual(2, c0.get_frame_count(frame_num))
        self.assertEqual(2, c0.get_frame_count(frame_num + 1))

