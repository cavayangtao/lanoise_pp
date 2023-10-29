#! /usr/bin/env python3

import rospy
import ros_numpy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import torch.nn.functional as F
from torch.utils import data
import torch
from dataset import RandlanetDataset
from sampler import RandlanetWeightedSampler
from model import RandlaNet
from hyperparameters import hyp
import time

def unpack_input(input_list, n_layers, device):
    inputs = dict()
    inputs['xyz'] = input_list[:n_layers]
    inputs['neigh_idx'] = input_list[n_layers: 2 * n_layers]
    inputs['sub_idx'] = input_list[2 * n_layers:3 * n_layers]
    inputs['interp_idx'] = input_list[3 * n_layers:4 * n_layers]
    for key, val in inputs.items():
        inputs[key] = [x.to(device) for x in val]
    inputs['features'] = input_list[4 * n_layers].to(device)
    inputs['labels'] = input_list[4 * n_layers + 1].to(device)
    inputs['input_inds'] = input_list[4 * n_layers + 2].to(device)
    inputs['cloud_inds'] = input_list[4 * n_layers + 3].to(device)
    return inputs

def convert_pc2ply(pc_label):
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    pc_label = pc_label.astype(np.float32)
    pc_label[:, 3:6] = pc_label[:, 3:6] + 1.1

    return pc_label, xyz_min

def segment(test_loader, model, device, cfg, max_epoch=1):
    n_points = len(test_loader.dataset)
    n_classes = model.n_classes
    xyz_probs = np.zeros((n_points, n_classes))
    xyz_probs[:] = np.nan
    test_smooth = 0.98
    with torch.no_grad():
        for step in range(max_epoch):
            print(f"Round {step}")

            for input_list in test_loader:

                inputs = unpack_input(input_list, cfg['num_layers'], device)
                outputs = model(inputs)                
                outputs = F.log_softmax(outputs, dim=1)
                outputs = torch.reshape(outputs, [cfg['val_batch_size'], -1, cfg['num_classes']])
                for j in range(outputs.shape[0]):
                    probs = outputs[j, :, :].cpu().detach().float().numpy()
                    ids = inputs['input_inds'][j, :].cpu().detach().int().numpy()
                    xyz_probs[ids] = np.nanmean([xyz_probs[ids], np.exp(probs)], axis=0)
                    # xyz_probs[ids] = test_smooth * xyz_probs[ids] \
                    #     + (1 - test_smooth) * probs
    
    for pc_id in test_loader.dataset.kdtrees:
        xyz_tile = test_loader.dataset.kdtrees[pc_id].data
    xyz_labels = np.argmax(xyz_probs, axis=1)

    return xyz_tile, xyz_labels

def callback(msg):
    
    points = ros_numpy.numpify(msg)
    pt_x = points['x']
    pt_y = points['y']
    pt_z = points['z']
    pt_i = np.zeros(len(pt_x)) + 1.1
    pt_l = np.zeros(len(pt_x))

    index = np.where(np.array(pt_z) <= top)
    if np.size(index[0]) > 0:
        pt_x = np.array(pt_x)[index[0]]
        pt_y = np.array(pt_y)[index[0]]
        pt_z = np.array(pt_z)[index[0]]
        pt_i = np.array(pt_i)[index[0]]
        pt_l = np.array(pt_l)[index[0]]
    else:
        pt_x = []
        pt_y = []
        pt_z = []
        pt_i = []
        pt_l = []

    pc_label = np.transpose(np.vstack((pt_x, pt_y, pt_z, pt_i, pt_i, pt_i, pt_l)))
    pc_label, xyz_min = convert_pc2ply(pc_label)

    test_params = {"batch_size": cfg['val_batch_size'],
                   "shuffle": False,
                   "num_workers": num_workers}  

    test_set = RandlanetDataset(pc_label, **cfg)
    test_sampler = RandlanetWeightedSampler(test_set,
                                            cfg['val_batch_size'] * cfg[
                                                'val_steps'])

    test_loader = data.DataLoader(test_set, sampler=test_sampler, **test_params)

    start = time.time()
    xyz_tile, xyz_labels = \
        segment(test_loader, model, device, cfg)
    end = time.time()
    print('Testing...', end - start)
    cloud_points = np.transpose(np.vstack((np.transpose(xyz_tile), xyz_labels)))
    cloud_points[:, 0:3] += xyz_min
    print(np.shape(cloud_points))

    index_good = np.where(cloud_points[:, 3] == 0)
    if np.size(index_good[0]) > 0:
        cloud_points = cloud_points[index_good[0], :]
    else:
        cloud_points = []

    #header
    header = std_msgs.msg.Header()
    header.stamp = msg.header.stamp
    header.frame_id = 'velodyne'
    #create pcl from points
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('intensity', 16, PointField.FLOAT32, 1),
          ]
    new_points = pc2.create_cloud(header, fields, cloud_points)
    pub.publish(new_points)

if __name__ == '__main__':
    bottom = -5
    top = 10
    cfg = hyp
    num_workers = 0

    rospy.init_node('lpp', anonymous=True)
    nice_model = rospy.get_param('~nice_model_path')
    print("Setting up pytorch")
    use_cuda = torch.cuda.is_available()
    print("Use cuda: ", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")   
    model = RandlaNet(n_layers=cfg['num_layers'], n_classes=cfg['num_classes'], d_out=cfg['d_out'])
    if not use_cuda:
        map_location = torch.device("cpu")
        model.load_state_dict(torch.load(nice_model, map_location=map_location))
    else:
        model.load_state_dict(torch.load(nice_model))

    # model.half
    model = model.to(device)
    model.eval()

    sub = rospy.Subscriber('velodyne_points', PointCloud2, callback)
    pub = rospy.Publisher('filtered_points', PointCloud2, queue_size=1)

    rospy.spin()