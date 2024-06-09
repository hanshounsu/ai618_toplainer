import os
import datetime
from pathlib import Path
import torch.nn as nn

def prepare_checkpoint_path(config):
    if not os.path.isdir(config.checkpoint_path_specific):
        if config.checkpoint_path_action == 'overwrite':
            os.makedirs(config.checkpoint_path_specific)
        elif config.checkpoint_path_action == 'duplicate':
            checkpoint_path = config.checkpoint_path_specific
            n = 0
            while os.path.exists(checkpoint_path):
                n += 1
                checkpoint_path = checkpoint_path + f'_{n}'
                os.makedirs(checkpoint_path)

def obtain_current_time():
    date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    date = date.strftime('%Y-%m-%d-%H-%M-%S')
    return date


def read_file_list(filename):
    with open(filename) as f:
        # make a list of each lines in file (no \n included)
        file_list = f.read().splitlines()
    return file_list

def obtain_list_of_filename(list_):
    return [os.path.basename(path_).replace('.wav','') for path_ in list_]

def write_file_list(dataset_path, config):
    pass
    # list_dataset = list(Path(f'{dataset_path}/wav').glob(f'*_{config.type_of_file[0]}.wav'))
    # train_list = list_dataset[:round(len(list_dataset)*0.9)]
    # val_list = list_dataset[round(len(list_dataset)*0.9):]

    # if config.split_type == 'all':
    #     train_txt = 'train_all.txt'
    #     test_txt = 'test_all.txt'
    # elif config.split_type == 'comp':
    #     train_txt = 'train_comp.txt'
    #     test_txt = 'test_comp.txt'

    # with open(os.path.join(dataset_path, train_txt), 'w') as f:
    #     for wav in train_list:
    #         f.write(str(wav))
    #         f.write('\n')
    # with open(os.path.join(dataset_path, test_txt), 'w') as f:
    #     for wav in val_list:
    #         f.write(str(wav))
    #         f.write('\n')


def check_test_files(config):
    if config.split_type == 'all':
        train_txt = 'train_all.txt'
        test_txt = 'test_all.txt'
    elif config.split_type == 'comp':
        train_txt = 'train_comp.txt'
        test_txt = 'test_comp.txt'

    if config.data_augmentation:
        train_list = obtain_list_of_filename(read_file_list(os.path.join(config.dataset_path_mix, train_txt)))
        test_list = obtain_list_of_filename(read_file_list(os.path.join(config.dataset_path_mix, test_txt)))
    elif not config.data_augmentation:
        train_list = obtain_list_of_filename(read_file_list(os.path.join(config.dataset_path_mix, train_txt)))
        test_list = obtain_list_of_filename(read_file_list(os.path.join(config.dataset_path_mix, test_txt)))

    for split, test_file in config.test_list:
        if test_file in test_list:
            assert split == 'test', f'Error inferring, {test_file} not in {split}!!!!!'
        elif test_file in train_list:
            assert split == 'train', f'Error inferring, {test_file} not in {split}!!!!!'
        else:
            assert split == 'test', f'Error inferring, {test_file} not in {split}!!!!!'

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
