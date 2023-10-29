from model.testing import segment_randlanet
from model.hyperparameters import hyp

segment_randlanet(model_path="data/saved_models/repo_example/",
                  pc_path="./data_prepare/Dataset/labels_fog_60/pc_id=1698068926703656911/",
                  cfg=hyp,
                  num_workers=4, 
                  segmentation_name='example')