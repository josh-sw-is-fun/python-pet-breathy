from pet_breathy.point import Point

import numpy as np

'''
- PatchCluster contains 1 or more patches
- PatchClusterFrame represents a single reading for that patch that belongs to some cluster

Frame num   1           2           3           ...     N
Cluster 1   Patch 2     Patch 3
            Patch 4     Patch 4
            Patch 9

Cluster 2   Patch 5     Patch 5     Patch 6
            Patch 6     Patch 6

Cluster ...
'''

class PatchClusterFrame:
    def __init__(self, center: Point, bpm: float, bpm_strength: float, frame_num: int):
        self.center = center.clone()
        self.bpm = bpm
        self.bpm_strength = bpm_strength
        self.frame_num = frame_num

class PatchClusterSummary:
    def __init__(self, begin_frame_num, end_frame_num, bpm_avg, bpm_strength_avg):
        self.begin_frame_num = begin_frame_num
        self.end_frame_num = end_frame_num
        self.bpm_avg = bpm_avg
        self.bpm_strength_avg = bpm_strength_avg
        
        self.num_frames = (self.end_frame_num - self.begin_frame_num) + 1
    
    def __str__(self):
        return f'begin: {self.begin_frame_num}, end: {self.end_frame_num}, bpm: {self.bpm_avg}, strength: {self.bpm_strength_avg}, frames: {self.num_frames}'

class PatchClusterAverages:
    def __init__(self, frame_num, patch_count, bpm_avg, bpm_strength_avg):
        self.frame_num = frame_num
        self.patch_count = patch_count
        self.bpm_avg = bpm_avg
        self.bpm_strength_avg = bpm_strength_avg

class PatchCluster:
    def __init__(self, cluster_id, radius_distance: float, bpm_threshold: float, frame: PatchClusterFrame):
        self.cluster_id = cluster_id
        self.radius_distance = radius_distance
        self.bpm_threshold = bpm_threshold
        self.curr_frame_num = frame.frame_num
        self.curr_center = frame.center
        self.bpm = frame.bpm
        self.begin_frame_num = self.curr_frame_num
        self.end_frame_num = self.curr_frame_num
        
        # Todo Could be made into a class
        self.bpm_min = 99999
        self.bpm_max = -99999
        self.bpm_strength_max = 0
        self._update_frame_stats(frame)
        
        # Key:      Frame num
        # Value:    list of PatchClusterFrame objects
        self.frame_history = {
            self.curr_frame_num: [ frame ]
        }

    def get_id(self):
        return self.cluster_id

    def get_bpm(self):
        return self.bpm

    def add_frame(self, frame: PatchClusterFrame):
        if add := self._can_add_to_cluster(frame):
            try:
                frames = self.frame_history[frame.frame_num]
            except KeyError:
                frames = [ ]
                self.frame_history[frame.frame_num] = frames
                self.curr_center = frame.center
            self.curr_frame_num = frame.frame_num
            self.end_frame_num = self.curr_frame_num
            self._update_frame_stats(frame)
            frames.append(frame)
        return add

    def frame_num_in_range(self, frame_num):
        return self.begin_frame_num <= frame_num <= self.end_frame_num

    def get_current_frame_num(self):
        return self.curr_frame_num

    def get_current_frame_count(self):
        return self.get_frame_count(self.curr_frame_num)

    def get_frame_count(self, frame_num):
        try:
            return len(self.frame_history[frame_num])
        except KeyError:
            return 0

    def get_current_center(self):
        return self.curr_center

    def get_center(self, frame_num):
        return self.get_frames(frame_num)[0].center

    def get_current_frames(self):
        return self.get_frames(self.curr_frame_num)
    
    def get_frames(self, frame_num):
        return self.frame_history[frame_num]

    def get_summary(self):
        bpm_avg = 0.0
        bpm_strength_avg = 0.0
        
        for frame_num, frames in self.frame_history.items():
            avg = self.get_frame_average(frame_num)
            bpm_avg += avg.bpm_avg
            bpm_strength_avg += avg.bpm_strength_avg
        
        bpm_avg /= len(self.frame_history)
        bpm_strength_avg /= len(self.frame_history)
        
        return PatchClusterSummary(
            self.begin_frame_num,
            self.end_frame_num,
            bpm_avg,
            bpm_strength_avg)

    def get_current_frame_averages(self):
        return self.get_frame_avg(self.curr_frame_num)

    def get_frame_average(self, frame_num):
        frames = self.get_frames(frame_num)
        frame_count = len(frames)
        bpm_avg = 0.0
        bpm_strength_avg = 0.0
        
        for frame in frames:
            bpm_avg += frame.bpm
            bpm_strength_avg += frame.bpm_strength
        
        bpm_avg /= frame_count
        bpm_strength_avg /= frame_count
        
        return PatchClusterAverages(
            frame_num,
            frame_count,
            bpm_avg,
            bpm_strength_avg)

    def intersects(self, other):
        if abs(self.bpm - other.bpm) <= self.bpm_threshold:
            if self._calc_dist(self.curr_center, other.curr_center) <= (self.radius_distance + other.radius_distance):
                return True
        return False

    def _update_frame_stats(self, frame):
        if frame.bpm < self.bpm_min:
            self.bpm_min = frame.bpm
        if frame.bpm > self.bpm_max:
            self.bpm_max = frame.bpm
        if frame.bpm_strength > self.bpm_strength_max:
            self.bpm_strength_max = frame.bpm_strength

    def _can_add_to_cluster(self, frame):
        if (frame.bpm - self.bpm_threshold) <= self.bpm <= (frame.bpm + self.bpm_threshold):
            if self._calc_dist(self.curr_center, frame.center) <= self.radius_distance:
                return True
        return False

    def _calc_dist(self, p0, p1):
        return np.sqrt((p1.y - p0.y)**2 + (p1.x - p0.x)**2)

class PatchClusterManager:
    def __init__(self, radius_distance: float, bpm_threshold: float):
        self.radius_distance = radius_distance
        self.bpm_threshold = bpm_threshold
        self.curr_frame_num = 0
        self.cluster_id = 0
        
        # Clusters
        # Old clusters or history clusters
        self.clusters = [ ]
        
        # Key:      Cluster ID
        # Value:    Cluster
        self.all_clusters = { }
        
        # Key:      Group ID
        # Value:    PatchClusterGroup
        self.groups = { }
        self.group_id = 0
        self.all_groups = { }

    def add_frame(self, frame: PatchClusterFrame):
        self.curr_frame_num = frame.frame_num
        frame_added = False
        for cluster in self.clusters:
            if cluster.add_frame(frame):
                frame_added = True
        
        if not frame_added:
            # Create new cluster with frame
            cluster = PatchCluster(
                self._gen_cluster_id(),
                self.radius_distance,
                self.bpm_threshold,
                frame)
            self.clusters.append(cluster)
            self.all_clusters[cluster.get_id()] = cluster
    
    def finished_adding_frames_for_frame_num(self):
        '''If a cluster has had new frames added to it, then carry it forward. Otherwise move the
        cluster out so we can still track its history.'''
        new_clusters = [ ]
        old_clusters = [ ]
        
        for cluster in self.clusters:
            if cluster.get_current_frame_count() > 0 and cluster.get_current_frame_num() == self.curr_frame_num:
                new_clusters.append(cluster)
            else:
                old_clusters.append(cluster)
        self.clusters = new_clusters
        
        for cluster in self.clusters:
            cluster_id = cluster.get_id()
            
            group_ids = set()
            
            #for top_level_cluster_id, sub_level_cluster_ids in groups.items():
            for group_id, group in self.groups.items():
                for group_cluster_id in group.cluster_ids:
                    group_cluster = self.all_clusters[group_cluster_id]
                    if cluster.intersects(group_cluster):
                        group_ids.add(group_id)
                        # This cluster has overlap with at least one cluster in this group
                        break
            
            if not group_ids:
                # This cluster does not overlap with any groups
                group_id = self._gen_cluster_group_id()
                group = PatchClusterGroup(group_id, cluster)
                self.groups[group_id] = group
                self.all_groups[group_id] = group
            elif len(group_ids) == 1:
                self.groups[group_ids.pop()].add_cluster(cluster)
            else:
                # This cluster overlaps with at least 2 or more groups
                # Merge the groups into one
                top_group = self.groups[group_ids.pop()]
                while group_ids:
                    group_id = group_ids.pop()
                    top_group.merge(self.groups[group_id])
                    del self.groups[group_id]
    
        for cluster in old_clusters:
            for group_id in list(self.groups.keys()):
                group = self.groups[group_id]
                group.remove_cluster(cluster)
                if not group.cluster_ids:
                    del self.groups[group_id]
    
    def _gen_cluster_id(self):
        cluster_id = self.cluster_id
        self.cluster_id += 1
        return cluster_id

    def _gen_cluster_group_id(self):
        group_id = self.group_id
        self.group_id += 1
        return group_id

class PatchClusterGroup:
    def __init__(self, group_id, cluster):
        self.group_id = group_id
        self.begin_frame_num = cluster.begin_frame_num
        self.end_frame_num = cluster.end_frame_num
        self.cluster_ids = {cluster.get_id()}
        self.all_cluster_ids = {cluster.get_id()}
        self.bpm = cluster.bpm
        
        self.bpm_min = 99999
        self.bpm_max = -99999
        self.bpm_strength_max = 0
        self._update_stats(cluster)
    
    def frame_num_in_range(self, frame_num):
        return self.begin_frame_num <= frame_num <= self.end_frame_num
    
    def add_cluster(self, cluster):
        self._update_stats(cluster)
        self._add_cluster_id(cluster.get_id())
    
    def remove_cluster(self, cluster):
        try:
            self.cluster_ids.remove(cluster.get_id())
        except KeyError:
            pass
    
    def merge(self, other_group):
        self._update_stats(other_group)
        for cluster_id in other_group.cluster_ids:
            self._add_cluster_id(cluster_id)
    
    def get_score(self):
        frames = self.end_frame_num - self.begin_frame_num
        #return frames #* self.bpm_strength_max
        return (frames, self.bpm_strength_max)
    
    def _add_cluster_id(self, cluster_id):
        self.cluster_ids.add(cluster_id)
        self.all_cluster_ids.add(cluster_id)
    
    def _update_stats(self, thing):
        if thing.begin_frame_num < self.begin_frame_num:
            self.begin_frame_num = thing.begin_frame_num
        if thing.end_frame_num > self.end_frame_num:
            self.end_frame_num = thing.end_frame_num
        
        if thing.bpm_min < self.bpm_min:
            self.bpm_min = thing.bpm_min
        if thing.bpm_max > self.bpm_max:
            self.bpm_max = thing.bpm_max
        if thing.bpm_strength_max > self.bpm_strength_max:
            self.bpm_strength_max = thing.bpm_strength_max

