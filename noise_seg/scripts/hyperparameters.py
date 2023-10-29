hyp = {
    'k_n': 16,  # KNN
    'num_layers': 5,  # Number of layers 5, 3
    'num_points': 10000,  # Number of input points
    'num_classes': 2,  # Number of valid classes
    'sub_grid_size': 0.04,  # preprocess_parameter
    'batch_size': 6,  # batch_size during training
    'val_batch_size': 1,  # batch_size during validation and test
    'train_steps': 500,  # Number of steps per epochs
    'val_steps': 5,  # Number of validation steps per epoch
    'sub_sampling_ratio': [4, 4, 4, 4, 2],  # sampling ratio of random sampling at each layer [4, 4, 4, 4, 2], [4, 4, 2]
    'd_out': [16, 64, 128, 256, 512],  # feature dimension [16, 64, 128, 256, 512], [64, 256, 512]
    'noise_init': 3.5,  # noise initial parameter
    'max_epoch': 100,  # maximum epoch during training
    'learning_rate': 1e-2,  # initial learning rate
    'lr_decays': {i: 0.95 for i in range(0, 500)},  # decay rate of learning rate
    'augment_scale_anisotropic': True,
    'augment_symmetries': [True, False, False],
    'augment_rotation': 'vertical',
    'augment_scale_min': 0.8,
    'augment_scale_max': 1.2,
    'augment_noise': 0.001,
    'augment_occlusion': 'none',
    'augment_color': 0.8
}
