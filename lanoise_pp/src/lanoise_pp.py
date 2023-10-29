#! /usr/bin/env python3

import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np

import scipy.special
from umal import umal
from umal import internal_network
import torch
import os

def ald_log_pdf(y, mu, b, tau):
    """
    Logarithm of the Asymmetric Laplace Probability density function
    """
    return np.where(
        y > mu,
        np.log(tau) + np.log(1 - tau) - np.log(b) - tau * (y - mu) / b,
        np.log(tau) + np.log(1 - tau) - np.log(b) - (tau - 1) * (y - mu) / b)

def callback(msg):

    pt_x = []
    pt_y = []
    pt_z = []
    pt_d = []
    pt_i = []
    pt_x_new1 = []
    pt_y_new1 = []
    pt_z_new1 = []
    pt_i_new1 = []
    pt_x_new2 = []
    pt_y_new2 = []
    pt_z_new2 = []
    pt_i_new2 = []

    # get all the points
    points = pc2.read_points(msg, field_names = ("x", "y", "z", "intensity"), skip_nans=False)
    for point in points:
        pt_x.append(point[0])
        pt_y.append(point[1])
        pt_z.append(point[2])
        pt_d.append(np.sqrt(np.square(point[0]) + np.square(point[1]) + np.square(point[2])))
        pt_i.append(point[3])

    pt_x = np.array(pt_x)
    pt_y = np.array(pt_y)
    pt_z = np.array(pt_z)
    pt_d = np.array(pt_d)
    pt_i = np.array(pt_i)
    print('maximal intensity: ', np.max(pt_i))
    print('median intensity: ', np.median(pt_i))

    # only deal with points in max_range
    index_outrange = np.where(pt_d > max_range)
    if np.size(index_outrange[0]) > 0:
        pt_x = np.delete(pt_x, index_outrange[0])
        pt_y = np.delete(pt_y, index_outrange[0])
        pt_z = np.delete(pt_z, index_outrange[0])
        pt_d = np.delete(pt_d, index_outrange[0])
        pt_i = np.delete(pt_i, index_outrange[0])

    # process the points for diffuse reflectors
    index = np.where(pt_i <= intensity)
    if np.size(index[0]) > 0:
        pt_x_new = pt_x[index[0]]
        pt_y_new = pt_y[index[0]]
        pt_z_new = pt_z[index[0]]
        pt_d_new = pt_d[index[0]]
        pt_i_new = pt_i[index[0]]

        # range prediction
        x_pred = np.vstack((np.ones(np.size(pt_i_new)) * visibility, pt_i_new, pt_d_new))
        x_pred = np.atleast_2d(x_pred).T
        x_pred[:, 0] = x_pred[:, 0] / 200.0
        x_pred[:, 1] = x_pred[:, 1] / 100.0
        x_pred[:, 2] = x_pred[:, 2] / 30.0

        nx = len(index[0])
        ny = 1000
        ntaus = 100
        x_synthetic = x_pred
        x_synthetic = torch.from_numpy(x_synthetic.astype('float32')).to(umal1_range.device)
        x_repeat = x_synthetic.view(nx, 1, umal1_range.n_dim).expand(nx, ntaus, umal1_range.n_dim)
        sel_taus = np.linspace(0. + 5e-2, 1. - 5e-2, ntaus)
        taus = np.tile(sel_taus[None, :, None], (nx, 1, 1))
        taus = torch.from_numpy(taus.astype('float32')).to(umal1_range.device)
        tmp_data = torch.cat([x_repeat, taus], dim=2)
        tmp_data = tmp_data.view(nx * ntaus, -1)
        with torch.no_grad():
            mu, b = umal1_range.model(tmp_data)
            mu = mu.cpu().numpy().reshape((nx, ntaus, 1))
            b = b.cpu().numpy().reshape((nx, ntaus, 1))
            taus = taus.cpu().numpy().reshape((nx, ntaus, 1))
            
        im = np.zeros((ny, nx))
        y = np.linspace(0, 50, ny)
        for i in range(ny):
            im[i, :] = scipy.special.logsumexp(ald_log_pdf(y[i], mu[:, :, 0], b[:, :, 0], taus[:, :, 0]),
                                               axis=1) - np.log(
                ntaus)
        pdf = np.exp(im)
        
        # sampling
        pt_d_new1_c = []
        pt_p_new1_c = []
        for index_x in range(pdf.shape[-1]):
            pdf_x = pdf[:, index_x]
            pdf_x = pdf_x / np.sum(pdf_x)
            sampled_distanse = np.random.choice(y, size=1, replace=True, p=pdf_x)
            pt_d_new1_c.append(sampled_distanse[0])
            index_p = np.where(y == sampled_distanse[0])
            pt_p_new1_c.append(pdf_x[index_p[0]])

        pt_p_new1_c = np.squeeze(pt_p_new1_c)
        pt_d_new1_c = np.array(pt_d_new1_c)
        index_small = np.where(pt_d_new1_c < mean_mu)
        small_noise = np.random.normal(mean_mu, 0.1, np.size(pt_d_new1_c[index_small[0]]))
        pt_d_new1_c[index_small[0]] = small_noise
        index_large = np.where(pt_d_new1_c > pt_d_new)
        pt_d_new1_c[index_large[0]] = pt_d_new[index_large[0]]
        ratio1 = pt_d_new1_c / pt_d_new
        pt_x_new1_c = pt_x_new * ratio1
        pt_y_new1_c = pt_y_new * ratio1
        pt_z_new1_c = pt_z_new * ratio1
        errors = abs(pt_d_new1_c - pt_d_new)
        index_vis = np.where((errors <= sigma_d) & (pt_p_new1_c >= 0.001))

        # get good points
        if np.size(index_vis[0]) > 0:
            pt_x_new1_a = pt_x_new[index_vis[0]]
            pt_y_new1_a = pt_y_new[index_vis[0]]
            pt_z_new1_a = pt_z_new[index_vis[0]]
            pt_i_new1_a = pt_i_new[index_vis[0]]
            # set intensity to 1
            pt_i_new1_a = np.zeros(np.size(pt_i_new1_a)) + 1

        # get noisy points
        if (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new1_c = np.delete(pt_x_new1_c, index_vis[0])
            pt_y_new1_c = np.delete(pt_y_new1_c, index_vis[0])
            pt_z_new1_c = np.delete(pt_z_new1_c, index_vis[0])
            pt_i_new1_c = np.delete(pt_i_new, index_vis[0])
            # set intensity to 1
            pt_i_new1_c = np.zeros(np.size(pt_i_new1_c)) + 250

        # put the good points and noisy points together
        if np.size(index_vis[0]) > 0 and (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new1 = np.hstack((pt_x_new1_a, pt_x_new1_c))
            pt_y_new1 = np.hstack((pt_y_new1_a, pt_y_new1_c))
            pt_z_new1 = np.hstack((pt_z_new1_a, pt_z_new1_c))
            pt_i_new1 = np.hstack((pt_i_new1_a, pt_i_new1_c))
        if np.size(index_vis[0]) > 0 and (np.size(pt_x_new) - np.size(index_vis[0])) == 0:
            pt_x_new1 = pt_x_new1_a
            pt_y_new1 = pt_y_new1_a
            pt_z_new1 = pt_z_new1_a
            pt_i_new1 = pt_i_new1_a
        if np.size(index_vis[0]) == 0 and (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new1 = pt_x_new1_c
            pt_y_new1 = pt_y_new1_c
            pt_z_new1 = pt_z_new1_c
            pt_i_new1 = pt_i_new1_c

    # process the points for retro-reflectors
    index = np.where(pt_i > intensity)
    if np.size(index[0]) > 0:
        pt_x_new = pt_x[index[0]]
        pt_y_new = pt_y[index[0]]
        pt_z_new = pt_z[index[0]]
        pt_d_new = pt_d[index[0]]
        pt_i_new = pt_i[index[0]]

        # range prediction
        x_pred = np.vstack((np.ones(np.size(pt_i_new)) * visibility, pt_i_new, pt_d_new))
        x_pred = np.atleast_2d(x_pred).T
        x_pred[:, 0] = x_pred[:, 0] / 200.0
        x_pred[:, 1] = x_pred[:, 1] / 100.0
        x_pred[:, 2] = x_pred[:, 2] / 30.0

        nx = len(index[0])
        ny = 1000
        ntaus = 100
        x_synthetic = x_pred
        x_synthetic = torch.from_numpy(x_synthetic.astype('float32')).to(umal2_range.device)
        x_repeat = x_synthetic.view(nx, 1, umal2_range.n_dim).expand(nx, ntaus, umal2_range.n_dim)
        sel_taus = np.linspace(0. + 5e-2, 1. - 5e-2, ntaus)
        taus = np.tile(sel_taus[None, :, None], (nx, 1, 1))
        taus = torch.from_numpy(taus.astype('float32')).to(umal2_range.device)
        tmp_data = torch.cat([x_repeat, taus], dim=2)
        tmp_data = tmp_data.view(nx * ntaus, -1)
        with torch.no_grad():
            mu, b = umal2_range.model(tmp_data)
            mu = mu.cpu().numpy().reshape((nx, ntaus, 1))
            b = b.cpu().numpy().reshape((nx, ntaus, 1))
            taus = taus.cpu().numpy().reshape((nx, ntaus, 1))

        im = np.zeros((ny, nx))
        y = np.linspace(0, 50, ny)
        for i in range(ny):
            im[i, :] = scipy.special.logsumexp(ald_log_pdf(y[i], mu[:, :, 0], b[:, :, 0], taus[:, :, 0]),
                                               axis=1) - np.log(
                ntaus)
        pdf = np.exp(im)

        # sampling
        pt_d_new2_c = []
        pt_p_new2_c = []
        for index_x in range(pdf.shape[-1]):
            pdf_x = pdf[:, index_x]
            pdf_x = pdf_x / np.sum(pdf_x)
            sampled_distanse = np.random.choice(y, size=1, replace=True, p=pdf_x)
            pt_d_new2_c.append(sampled_distanse[0])
            index_p = np.where(y == sampled_distanse[0])
            pt_p_new2_c.append(pdf_x[index_p[0]])

        pt_p_new2_c = np.squeeze(pt_p_new2_c)
        pt_d_new2_c = np.array(pt_d_new2_c)
        index_small = np.where(pt_d_new2_c < mean_mu)
        small_noise = np.random.normal(mean_mu, 0.1, np.size(pt_d_new2_c[index_small[0]]))
        pt_d_new2_c[index_small[0]] = small_noise
        index_large = np.where(pt_d_new2_c > pt_d_new)
        pt_d_new2_c[index_large[0]] = pt_d_new[index_large[0]]
        ratio2 = pt_d_new2_c / pt_d_new
        pt_x_new2_c = pt_x_new * ratio2
        pt_y_new2_c = pt_y_new * ratio2
        pt_z_new2_c = pt_z_new * ratio2
        errors = abs(pt_d_new2_c - pt_d_new)
        index_vis = np.where((errors <= sigma_r) & (pt_p_new2_c >= 0.001))

        # get good points
        if np.size(index_vis[0]) > 0:
            pt_x_new2_a = pt_x_new[index_vis[0]]
            pt_y_new2_a = pt_y_new[index_vis[0]]
            pt_z_new2_a = pt_z_new[index_vis[0]]
            pt_i_new2_a = pt_i_new[index_vis[0]]
            # set intensity to 1
            pt_i_new2_a = np.zeros(np.size(pt_i_new2_a)) + 1

        # get noisy points
        if (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new2_c = np.delete(pt_x_new2_c, index_vis[0])
            pt_y_new2_c = np.delete(pt_y_new2_c, index_vis[0])
            pt_z_new2_c = np.delete(pt_z_new2_c, index_vis[0])
            pt_i_new2_c = np.delete(pt_i_new, index_vis[0])
            # set intensity to 1
            pt_i_new2_c = np.zeros(np.size(pt_i_new2_c)) + 250

        # put the good points and noisy points together
        if np.size(index_vis[0]) > 0 and (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new2 = np.hstack((pt_x_new2_a, pt_x_new2_c))
            pt_y_new2 = np.hstack((pt_y_new2_a, pt_y_new2_c))
            pt_z_new2 = np.hstack((pt_z_new2_a, pt_z_new2_c))
            pt_i_new2 = np.hstack((pt_i_new2_a, pt_i_new2_c))
        if np.size(index_vis[0]) > 0 and (np.size(pt_x_new) - np.size(index_vis[0])) == 0:
            pt_x_new2 = pt_x_new2_a
            pt_y_new2 = pt_y_new2_a
            pt_z_new2 = pt_z_new2_a
            pt_i_new2 = pt_i_new2_a
        if np.size(index_vis[0]) == 0 and (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new2 = pt_x_new2_c
            pt_y_new2 = pt_y_new2_c
            pt_z_new2 = pt_z_new2_c
            pt_i_new2 = pt_i_new2_c

    # put all the points together
    if np.size(pt_x_new1) > 0:
        cloud_points1 = np.transpose(np.vstack((pt_x_new1, pt_y_new1, pt_z_new1, pt_i_new1)))
        # cloud_points1 = np.transpose(np.vstack((pt_x_new1, pt_y_new1, pt_z_new1)))
    if np.size(pt_x_new2) > 0:
        cloud_points2 = np.transpose(np.vstack((pt_x_new2, pt_y_new2, pt_z_new2, pt_i_new2)))
        # cloud_points2 = np.transpose(np.vstack((pt_x_new2, pt_y_new2, pt_z_new2)))

    if np.size(pt_x_new1) > 0 and np.size(pt_x_new2) > 0:
        cloud_points = np.vstack((cloud_points1, cloud_points2))
    if np.size(pt_x_new1) > 0 and np.size(pt_x_new2) == 0:
        cloud_points = cloud_points1
    if np.size(pt_x_new1) == 0 and np.size(pt_x_new2) > 0:
        cloud_points = cloud_points2

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('intensity', 12, PointField.FLOAT32, 1)
          ]

    #header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'velodyne'
    #create pcl from points
    new_points = pc2.create_cloud(header, fields, cloud_points)
    pub.publish(new_points)
    # rospy.loginfo(points)

    # generate the labels of good and noisy points
    if save_data:
        creat_path = '/home/tyang/data/labels_fog_'+ str(visibility) + '/' + 'boards_' + str(header.stamp)
        isExists = os.path.exists(creat_path)
        if not isExists:
            points_path = os.makedirs(creat_path)
            labels_path = os.makedirs(creat_path + '/Annotations')

        points_all = np.column_stack((cloud_points, cloud_points[:, 3], cloud_points[:, 3]))
        file_name = 'boards_' + str(header.stamp) + '.txt'
        np.savetxt(creat_path + '/' + file_name, points_all, delimiter = ' ', fmt='%.3f')

        if np.size(pt_i_new1_c) > 0:
            points_noise1 = np.transpose(np.vstack((pt_x_new1_c, pt_y_new1_c, pt_z_new1_c, pt_i_new1_c)))
        if np.size(pt_i_new2_c) > 0:
            points_noise2 = np.transpose(np.vstack((pt_x_new2_c, pt_y_new2_c, pt_z_new2_c, pt_i_new2_c)))
        if np.size(pt_i_new1_c) > 0 and np.size(pt_i_new2_c) > 0:
            points_noise = np.vstack((points_noise1, points_noise2))
        if np.size(pt_i_new1_c) > 0 and np.size(pt_i_new2_c) == 0:
            points_noise = points_noise1
        if np.size(pt_i_new1_c) == 0 and np.size(pt_i_new2_c) > 0:
            points_noise = points_noise2
        points_noise = np.column_stack((points_noise, points_noise[:, 3], points_noise[:, 3]))
        file_name = 'noise_' + str(header.stamp) + '.txt'
        np.savetxt(creat_path + '/Annotations/' + file_name, points_noise, delimiter = ' ', fmt='%.3f')

        if np.size(pt_i_new1_a) > 0:
            points_noise1 = np.transpose(np.vstack((pt_x_new1_a, pt_y_new1_a, pt_z_new1_a, pt_i_new1_a)))
        if np.size(pt_i_new2_a) > 0:
            points_noise2 = np.transpose(np.vstack((pt_x_new2_a, pt_y_new2_a, pt_z_new2_a, pt_i_new2_a)))
        if np.size(pt_i_new1_a) > 0 and np.size(pt_i_new2_a) > 0:
            points_noise = np.vstack((points_noise1, points_noise2))
        if np.size(pt_i_new1_a) > 0 and np.size(pt_i_new2_a) == 0:
            points_noise = points_noise1
        if np.size(pt_i_new1_a) == 0 and np.size(pt_i_new2_a) > 0:
            points_noise = points_noise2
        points_noise = np.column_stack((points_noise, points_noise[:, 3], points_noise[:, 3]))
        file_name = 'good_' + str(header.stamp) + '.txt'
        np.savetxt(creat_path + '/Annotations/' + file_name, points_noise, delimiter = ' ', fmt='%.3f')

if __name__ == '__main__':

    visibility = 60  
    save_data = True
    sigma_d = 0.5 # set the threshold for good points
    sigma_r = 1.0 # set the threshold for good points
    intensity = 100 # set the intensity threshold to distinguish diffuse reflectors and retro-reflectors
    max_range = 35 # maximal simulation range

    np.random.seed(1)
    rospy.init_node('lanoise_pp', anonymous=True)

    umal1_range_path = rospy.get_param('~umal1_range_path')
    umal2_range_path = rospy.get_param('~umal2_range_path')
    umal1_range = umal(n_dim=3)
    umal1_range.init_prediction(architecture=internal_network)
    umal1_range.load_weights(umal1_range_path)
    umal2_range = umal(n_dim=3)
    umal2_range.init_prediction(architecture=internal_network)
    umal2_range.load_weights(umal2_range_path)

    # range prediction for minimal prediction
    x_pred = np.vstack((5, 60, 25))
    x_pred = np.atleast_2d(x_pred).T
    x_pred[:, 0] = x_pred[:, 0] / 200.0
    x_pred[:, 1] = x_pred[:, 1] / 100.0
    x_pred[:, 2] = x_pred[:, 2] / 30.0

    nx = 1
    ny = 1000
    ntaus = 100
    x_synthetic = x_pred
    x_synthetic = torch.from_numpy(x_synthetic.astype('float32')).to(umal1_range.device)
    x_repeat = x_synthetic.view(nx, 1, umal1_range.n_dim).expand(nx, ntaus, umal1_range.n_dim)
    sel_taus = np.linspace(0. + 5e-2, 1. - 5e-2, ntaus)
    taus = np.tile(sel_taus[None, :, None], (nx, 1, 1))
    taus = torch.from_numpy(taus.astype('float32')).to(umal1_range.device)
    tmp_data = torch.cat([x_repeat, taus], dim=2)
    tmp_data = tmp_data.view(nx * ntaus, -1)
    with torch.no_grad():
        mu, b = umal1_range.model(tmp_data)
        mu = mu.cpu().numpy().reshape((nx, ntaus, 1))
        b = b.cpu().numpy().reshape((nx, ntaus, 1))
        taus = taus.cpu().numpy().reshape((nx, ntaus, 1))
    mean_mu = np.mean(mu[:, :, 0], axis = 1)
    print('minimal distance: ', mean_mu)

    sub = rospy.Subscriber('velodyne_points', PointCloud2, callback)
    pub = rospy.Publisher('filtered_points', PointCloud2, queue_size=1)

    rospy.spin()
