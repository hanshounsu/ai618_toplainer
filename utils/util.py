import librosa
import torch
import shutil
import os

def obtain_start_end_of_mute(filepath, REF=0.1, threshold=32000):
    mute_start_end = []
    audio, sr = librosa.load(filepath, sr=16000, mono=True)
    aligned_retain = torch.zeros((len(audio),))
    state=0
    temp_time=0
    for idx, val in enumerate(audio):
        if idx == len(audio)-1:
            if state == 0:
                if idx - temp_time > threshold:
                    aligned_retain[temp_time:] = -1
                    mute_start_end.append((temp_time, idx)) # start and end
        else:
            if val > REF:
                if state == 0:
                    state = 1
                    if idx-temp_time > threshold:
                        aligned_retain[temp_time:idx] = -1
                        mute_start_end.append((temp_time, idx)) # start and end
                    temp_time = 0
                    
            elif val <= REF:
                if state == 1:
                    state = 0
                    temp_time = idx
    return (filepath, audio, aligned_retain, mute_start_end)


def make_groundtruth_evaluation_set(config):
    for split, filename in config.test_list:
        guitar_path = os.path.join(config.dataset_path, filename+'.wav')
        vocal_path = os.path.join(config.dataset_path.replace('/guitar', '/vocal'), filename+'.wav')
        target_path = vocal_path.replace('/segments', '/save/groundtruth').replace('/vocal', '/vocal/evaluation')
        if not os.path.exists(target_path):
            shutil.copy(vocal_path, target_path)