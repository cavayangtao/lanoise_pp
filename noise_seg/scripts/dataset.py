import random
from collections import defaultdict
import torch
import numpy as np
from scipy.spatial import cKDTree
from torch.utils import data
from utils import read_metadata, rotate

class RandlanetDataset(data.Dataset):

    def __init__(self, pc_label, **kwargs):
        self.cfg = kwargs
        self.size = 0
        pc_labels = [0, 1]
        self.test = [-99.] == pc_labels
        self.mapping = {0: 0, 1: 1}
        self.kdtrees = dict()
        self.colors = dict()
        self.labels = dict()
        self.pc_class_count = dict()
        self.total_class_count = defaultdict(int)
        self.total_class_weight = dict()
        self.n_points = 0

        pc = pc_label
        pc_id = 0
        kdtree = cKDTree(pc[:, :3], leafsize=50)
        self.kdtrees[pc_id] = kdtree
        self.colors[pc_id] = pc[:, 3:6]/255.
        self.labels[pc_id] = pc[:, 6]
        self.size += len(self.kdtrees[pc_id].data)

        labels, counters = np.unique(self.labels[pc_id], return_counts=True)
        self.pc_class_count[pc_id] = dict()
        for label, counter in zip(labels, counters):
            self.pc_class_count[pc_id][label] = counter
            self.total_class_count[label] += counter
            self.n_points += counter

        for label, counter in self.total_class_count.items():
            self.total_class_weight[label] = counter/self.n_points

    def __getitem__(self, _tuple):
        pc_id = _tuple[0]
        pick_point = _tuple[1]
        # center_point = _tuple[1].reshape(1, -1)
        # Get all points within the cloud from tree structure
        points = np.array(self.kdtrees[pc_id].data, copy=False)

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < self.cfg['num_points']:
            # Query all points within the cloud
            query_idx = self.kdtrees[pc_id].query(pick_point,
                                                  k=len(points))[1][0]
        else:
            # Query the predefined number of points
            query_idx = self.kdtrees[pc_id].query(pick_point,
                                                  k=self.cfg['num_points'])[1][0]

        # shuffle index inplace
        random.shuffle(query_idx)

        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[query_idx]

        queried_pc_xyz[:, 0:3] = queried_pc_xyz[:, 0:3] - pick_point[:, 0:3]
        queried_pc_colors = self.colors[pc_id][query_idx]
        queried_pc_labels = self.labels[pc_id][query_idx]

        queried_pc_labels = np.array(
            [self.mapping[lbl] for lbl in queried_pc_labels])

        input_list = self.build_input(queried_pc_xyz, queried_pc_colors,
                                      queried_pc_labels, query_idx, pc_id)

        return input_list

    def __len__(self):
        return self.size

    def build_input(self, xyz, rgb, labels, query_idx, pc_id):
        features = torch.tensor(self.augment_input(xyz, rgb), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        query_idx = torch.tensor(query_idx, dtype=torch.int32)
        pc_id = torch.tensor(pc_id, dtype=torch.int32)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(self.cfg['num_layers']):
            _, neigh_idx = cKDTree(xyz).query(xyz, k=self.cfg['k_n'])
            sub_sampling_idx = len(xyz)//self.cfg['sub_sampling_ratio'][i]
            sub_points = xyz[:sub_sampling_idx]
            pool_i = neigh_idx[:sub_sampling_idx]
            _, up_i = cKDTree(sub_points).query(xyz, k=1)
            input_points.append(torch.tensor(xyz, dtype=torch.float32))
            input_neighbors.append(torch.tensor(neigh_idx, dtype=torch.int32))
            input_pools.append(torch.tensor(pool_i, dtype=torch.int32))
            input_up_samples.append(torch.tensor(up_i, dtype=torch.int32))
            xyz = sub_points

        inputs = input_points + input_neighbors + input_pools + input_up_samples
        inputs += [features, labels, query_idx, pc_id]
        return inputs

    def augment_input(self, xyz, rgb):
        theta = np.random.uniform(0.0, 2 * np.pi)
        transformed_xyz = rotate(xyz, [0., 0., theta])

        # Choose random scales for each example
        min_s = self.cfg['augment_scale_min']
        max_s = self.cfg['augment_scale_max']
        if self.cfg['augment_scale_anisotropic']:
            scales = np.random.uniform(min_s, max_s, size=(3,))
        else:
            scales = np.random.uniform(min_s, max_s)
            scales = np.array([scales, scales, scales])

        symmetries = []
        for i in range(3):
            if self.cfg['augment_symmetries'][i]:
                symmetries.append(np.round(
                    np.random.uniform()) * 2 - 1)
            else:
                symmetries.append(1.)
        scales *= np.array(symmetries)

        # Apply scales
        transformed_xyz = transformed_xyz * scales

        noise = np.random.normal(scale=self.cfg['augment_noise'],
                                 size=transformed_xyz.shape)
        transformed_xyz = transformed_xyz + noise

        stacked_features = np.concatenate([transformed_xyz, rgb], axis=-1)
        return stacked_features
