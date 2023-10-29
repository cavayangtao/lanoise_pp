from os.path import join, exists, dirname
import numpy as np
import pandas as pd
import os, glob, pickle
import shutil

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("New folder" + path)
    else:
        print("There is this folder!")

def convert_pc2ply(anno_path, save_path, pc_id):
    """
    Convert original dataset files to ply file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    :param anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
    :param save_path: path to save original point clouds (each line is XYZRGBL)
    :return: None
    """
    data_list = []

    for f in glob.glob(join(anno_path, '*.txt')):
        class_name = os.path.basename(f).split('_')[0]
        if class_name not in gt_class:  # note: in some room there is 'staris' class..
            class_name = 'clutter'
        pc = pd.read_csv(f, header=None, delim_whitespace=True).values
        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
        data_list.append(np.concatenate([pc, labels], 1))  # Nx7

    pc_label = np.concatenate(data_list, 0)
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    pc_label = pc_label.astype(np.float32)
    pc_label[:, 3:6] = pc_label[:, 3:6] + 0.1

    mkdir(save_path)
    write_pc = open(save_path + '/pc.pickle', 'wb')
    pickle.dump(pc_label, write_pc)
    write_pc.close()

    mkdir(save_path + '/metadata')
    write_meta = open(save_path + '/metadata/metadata.pickle', 'wb')
    metadata = {'pc_id': int(pc_id), 'labels': [0, 1], 'name': None}
    pickle.dump(metadata, write_meta)
    write_meta.close()

if __name__ == '__main__':

    file_name = 'labels_fog_60'
    # 在这里更改保存
    ply_path = './Dataset/' + file_name
    if os.path.isdir(ply_path):
        shutil.rmtree(ply_path)
        print('ply_path is removed')
    dataset_path = './Dataset/Dataset_Gen'
    anno_paths = [line.rstrip() for line in open('./meta/anno_paths.txt')]
    anno_paths = [join(dataset_path, p) for p in anno_paths]
    gt_class = [x.rstrip() for x in open('./meta/class_names.txt')]
    gt_class2label = {cls: i for i, cls in enumerate(gt_class)}
    original_pc_folder = join(dirname(dataset_path), file_name)
    os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
    out_format = '.pickle'

    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for annotation_path in anno_paths:
        print(annotation_path)
        elements = str(annotation_path).split('/')
        pc_id = elements[-2].split('_')
        pc_id = pc_id[-1]
        out_file_name = 'pc_id=' + pc_id
        convert_pc2ply(annotation_path, join(original_pc_folder, out_file_name), pc_id)
