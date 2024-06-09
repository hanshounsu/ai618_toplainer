import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from random import randint


class TrainDataset(Dataset): # 리턴값 (vocal_audio, guitar_audio)
    '''
    Dataset form : [[vocal_aud1, vocal_aud2, ..., vocal_audN], [guitar_aud1, guitar_aud2, ..., guitar_audN]]
    우선 곡부터 고르고, 그 곡 안에서 랜덤하게 뽑는 방식 (시도 중)
    '''

    def __init__(self, config, mode='train'):
        # 현수님의 pt 파일명에 따라 수정될 수 있음.
        pt = mode + '_norm.pt'
        
        # feature_path_specific이 configs 중에 어느 yml에 있는거지..? default.yml 혹은 diffusion.yml인 것 같은데 안 보인다. (feature_path는 있음)
        self.data = torch.load(os.path.join(config.feature_path_specific, pt), map_location=config.device) # note and spec
        self.config = config
        self.audio_length = config.audio_length # 5.12*16000=81920
        self.num_file = len(self.data[0])

    def __len__(self):
        self.length = 0
        for i in range(self.num_file):
            self.length += self.data[0][i].size(-1) - (self.audio_length)
        return self.length

    def __getitem__(self, idx):
        idx = randint(0, self.num_file - 1)
        idx2 = randint(0, self.data[0][idx].size(-1) - self.audio_length) # 유효한 곡의 범위를 얻어내는 과정
        # idx2는 시작 timestep를 의미함.
        
        vocal_audio = self.data[0][idx][idx2:idx2+self.audio_length]
        guitar_audio = self.data[1][idx][idx2:idx2+self.audio_length]

        #merged_list = [(vocal_audio[i], guitar_audio[i]) for i in range(0, len(vocal_audio))]
        #[(vocal_aud1, guitar_aud1) , (vocal_aud2, guitar_aud2), ..., (vocal_audN, guitar_audN)] 
        
        return (vocal_audio, guitar_audio)


class InferDataset(Dataset):
    '''
    Dataset form (tensor) : 6 x L
    data : output MIDI note of preprocessing
    '''

    def __init__(self, guitar_audio, config):
        self.audio_length = config.audio_length
        self.guitar_audio = guitar_audio
        self.inpainting_ratio = config.inpainting_ratio
        self.inpainting_length = int((config.inpainting_ratio)*self.audio_length)
        #self.length = (self.guitar_audio.size(-1)//(self.audio_length))*2 - 1
        self.length = 1+ (self.guitar_audio.size(-1) - self.audio_length) // self.inpainting_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        audio = self.guitar_audio[int(self.inpainting_ratio*idx*self.audio_length):int((self.inpainting_ratio*(idx)+1)*self.audio_length)]
        audio = audio.to(torch.float32)

        return audio


def load_train(config):
    dataset = TrainDataset(config, mode='train')
    print(f'Train dataset number : {dataset.__len__()}')
    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, drop_last=True, num_workers=config.num_proc_train)
    return dataloader



def load_test(config):
    dataset = TrainDataset(config, mode='test')
    print(f'Test dataset number : {dataset.__len__()}')
    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, drop_last=True, num_workers=config.num_proc_train)
    return dataloader


def load_infer(guitar_audio, config):
    dataset = InferDataset(guitar_audio, config)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, drop_last=True)
    return dataloader
