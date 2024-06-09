import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchcrepe
from torchaudio.transforms import Resample, Spectrogram
from librosa import resample
from pytsmod.wsolatsm import wsola

from extension.torch_xt import set_device
import math

# Scale Methods
def time2frame(x, frame_rate):
    return int(x*frame_rate)

def frame(x, win_size, hop_size):
    if x.dim() == 1:
        num_frame = (x.size(0) - win_size)//hop_size + 1
        y = x.new_zeros(win_size, num_frame)
        for i in range(num_frame):
            y[:,i] = x[hop_size*i:hop_size*i + win_size]
    elif x.dim() == 2:
        num_frame = (x.size(1) - win_size)//hop_size + 1
        y = x.new_zeros(x.size(0), win_size, num_frame)
        for i in range(x.size(0)):
            for j in range(num_frame):
                y[i,:,j] = x[i, hop_size*j:hop_size*j + win_size]
    else:
        raise AssertionError("Input dimension should be 1 or 2")

    return y

def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)

    return x 

def amp2db(x, min_level_db=None):
    x = to_tensor(x)

    x = 20.0*torch.log10(x)
    if min_level_db is not None:
        min_level_db = torch.tensor(min_level_db).float()
        x = torch.max(x, min_level_db)

    return x 

def db2amp(x):
    x = to_tensor(x)

    return torch.pow(10.0, x*0.05)

def hz2midi(x):
    x = to_tensor(x)
    x = 69.0 + 12.0*torch.log2(x/440.0)
    x[x == -float('inf')] = 0

    return x

def midi2hz(x):
    x = to_tensor(x)

    return 440.0*torch.pow(2.0, (x - 69)/12)

def normalize(x, max_x=None, preemphasis=1.0):
    if max_x == None:
        max_x = torch.max(x)
    x = torch.pow(x/max_x, preemphasis)
    x = torch.clamp(x, 0, 1)

    return x, max_x

def denormalize(x, max_x, preemphasis=1.0, deemphasis=1.0):
    x = torch.pow(x, deemphasis/preemphasis)
    x = max_x*x

    return x

def db_normalize(x, min_level_db):
    return torch.clamp((x - min_level_db)/(-min_level_db), 0, 1)

def db_denormalize(x, min_level_db):
    return x.clamp(0, 1)*(-min_level_db) + min_level_db

def db2log10std(x, config, device):
    mean, std = set_device(load_norm(config.norm_file), device)
    if config.use_ref_level:
        x = (db_denormalize(x, config.min_level_db) + config.ref_level_db)/20.0
    else:
        x = db_denormalize(x, config.min_level_db)/20.0
    
    if x.size(-1) != mean.size(0):
        x = x.transpose(-2, -1)
        x = (x - mean)/std
        x = x.transpose(-2, -1)
    else:
        x = (x - mean)/std

    return x

def linear2log10std(x, config, device, eps=1e-10):
    mean, std = set_device(load_norm(config.norm_file), device)
    max_x = db2amp(config.ref_level_db)
    eps = torch.tensor(eps)
    max_x, eps = set_device((max_x, eps), config.transform_device)

    x = denormalize(x, max_x, config.preemphasis, config.deemphasis)
    x = torch.log10(torch.max(eps, x))

    if x.size(-1) != mean.size(0):
        x = x.transpose(-2, -1)
        x = (x - mean)/std
        x = x.transpose(-2, -1)
    else:
        x = (x - mean)/std

    return x



def ln_normalize_hawthorne(spectrogram, config):
    clip_value_min_log = math.log(config.clip_value_min)
    clip_value_max_log = math.log(config.clip_value_max)
    if config.model_name == 'FFT_diffusion':
        spectrogram = (spectrogram- clip_value_min_log)/(clip_value_max_log-clip_value_min_log) * 2 - 1 # between -1 and 1
    elif config.model_name == 'FFT_conv':
        spectrogram = (spectrogram- clip_value_min_log)/(clip_value_max_log-clip_value_min_log) # between 0 and 1
    else:
        raise ValueError
        
    return spectrogram


def ln_denormalize_hawthorne(spectrogram, config):
    clip_value_min_log = math.log(config.clip_value_min)
    clip_value_max_log = math.log(config.clip_value_max)
    if config.model_name == 'FFT_diffusion':
        spectrogram = (spectrogram + 1)/2 * (clip_value_max_log-clip_value_min_log) + clip_value_min_log # clip_value_max_log and clip_value_min_log
    elif config.model_name == 'FFT_conv':
        spectrogram = spectrogram * (clip_value_max_log-clip_value_min_log) + clip_value_min_log # between 0 and 1
    else:
        raise ValueError

    return spectrogram

    


def load_norm(filename):
    mean, std = torch.from_numpy(np.load(filename))

    return mean, std

# Wave Methods
def load(filename, sample_rate, load_mono=False):
    y, source_rate = torchaudio.load(filename)
    if y.size(0) > 1 and load_mono:
        y = y[0].unsqueeze(0)

    if source_rate != sample_rate:
        resample = Resample(source_rate, sample_rate)
        y = resample(y)

    return y

def save(filename, wave, sample_rate):
    wave = torch.clamp(wave, -1, 1)
    wave = wave.cpu()
    torchaudio.save(filename, wave, sample_rate)

def rmse(y, win_size, hop_size, center=True, pad_mode='reflect', squeeze=True):
    if center:
        y = y.unsqueeze(0)
        y = F.pad(y, (win_size//2, win_size//2), pad_mode, 0)
        y = y.squeeze(0)

    y = frame(y, win_size, hop_size)
    rmse = torch.sqrt(torch.mean(torch.abs(y)**2, dim=y.dim() - 2))
    if squeeze:
        rmse = rmse.squeeze(0)

    return rmse

def pitch(y, sample_rate, model='full', step_size=10, scale='midi', discrete=True):
    hop_length = int((step_size/1000)*sample_rate)
    pitch, harmonicity = torchcrepe.predict(y, sample_rate, hop_length, model=model, return_harmonicity=True)
    harmonicity = torchcrepe.filter.median(harmonicity, 3)
    pitch[harmonicity < 0.21] = 0
    pitch = torchcrepe.filter.mean(pitch, 3)

    if scale == 'midi':
        pitch = hz2midi(pitch)
    if discrete:
        pitch = pitch.round().long()

    torch.cuda.empty_cache()

    return pitch

def wsola_pitch_shift(x, sample_rate, pitch_shift):
    beta = np.power(2.0, pitch_shift/12.0)
    ndim = x.ndim

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]

    x_resample = resample(x, sample_rate*beta, sample_rate)
    x_wsola = wsola(x_resample, beta)
    x_wsola = to_tensor(x_wsola)

    if ndim != x.ndim:
        x_wsola = x_wsola.unsqueeze(0)

    return x_wsola

# Spectral Methods
def stft(y, config): 
    spec_fn = Spectrogram(n_fft=config.fft_size, 
                          win_length=config.win_size, 
                          hop_length=config.hop_size)
    y, spec_fn = set_device((y, spec_fn), config.transform_device)
    spec = torch.sqrt(spec_fn(y))

    return spec

def istft(magnitude, phase, config):
    window = torch.hann_window(config.win_size)
    stft_matrix = torch.stack((magnitude*torch.cos(phase), magnitude*torch.sin(phase)), dim=-1)
    stft_matrix, window = set_device((stft_matrix, window), config.transform_device)
    y = torchaudio.functional.istft(stft_matrix,
                                    n_fft=config.fft_size,
                                    hop_length=config.hop_size,
                                    win_length=config.win_size,
                                    window=window)

    return y