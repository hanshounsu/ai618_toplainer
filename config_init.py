"""
Initializing config class before training.
Adding some variables to configs that depends on initial yml settings.
"""
import os
from pathlib import Path

def config_initialize(config):

    feature_path = os.path.join(config.feature_path, config.singer + '_' + config.model_name + '_' + config.exp_name)
    config.add('feature_path_specific', feature_path)

    checkpoint_path = os.path.join(config.checkpoint_path, config.singer + '_' + config.model_name + '_' + config.exp_name)
    config.add('checkpoint_path_specific', checkpoint_path)

    sample_save_path = os.path.join(config.sample_save_path, config.singer + '_' + config.model_name + '_' + config.exp_name + '/' + 'noise_' + str(config.adding_noise) + '_inpainting_ratio_' + str(config.inpainting_ratio))
    config.add('sample_save_path_specific', sample_save_path)


    if config.test_list==False:
        test_path = os.path.join(config.feature_path.replace('/inpainting', ''), 'mackenzie_test_list_vocal_norm.txt')
        with open(test_path, 'r') as f:
            test_list = f.read().splitlines()
        test_list = [['test', str(Path(x).stem)] for x in test_list]
        config.add('test_list', test_list)