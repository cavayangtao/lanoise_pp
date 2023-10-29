from model.hyperparameters import hyp
from model.training import train_randlanet_model
import os

training_path = ['./data_prepare/Dataset/labels_fog_60/']
test_path = ['./data_prepare/Dataset/labels_fog_60/']
training_set = []
test_set = []
for current_path in training_path:
    training_list = os.listdir(current_path)
    for current_list in training_list:
        training_set.append(current_path + current_list + '/')
for current_path in test_path:
    test_list = os.listdir(current_path)
    for current_list in test_list:
        test_set.append(current_path + current_list + '/')
print('Data paths are set.')

train_randlanet_model(train_set_list = training_set,
                      test_set_list = test_set,
                      hyperpars=hyp,
                      use_mlflow=False,
                      num_workers=4,
                      model_name="repo_example")