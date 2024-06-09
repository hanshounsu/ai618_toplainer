"""
Integrated logger.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.dirname(__file__)))))
import extension.dsp_xt as dsp
import matplotlib.pyplot as plt
import librosa.display
import librosa
from torchvision.utils import save_image
import wandb
from pathlib import Path


def wandblog(y_axis_name: str, y_val: float, x_axis_name: str, x_val: float) -> None:
    wandb.log({y_axis_name: y_val, x_axis_name: x_val})


def wandblogGAudio(wave, split, file_name, step, config):
    wave = wave.squeeze(0)
    wave = wave.cpu().numpy()  # shape -> (L,)
    wave_log = wandb.Audio(
        wave, sample_rate=config.sample_rate, caption=str(step))
    wandb.log({f'{str(split)}-{str(file_name)}': wave_log})


def logTensor(y_axis_name, tensor, step=None, type='wandb'):
    if type=='wandb':
        # Tensor is attention matrix
        images = wandb.Image(tensor, caption=f"step: {step}", mode='L')
        wandb.log({f"{y_axis_name}": images})

    
def wandblogEEM(y_axis_name, tensor, step):
    # Tensor is attention matrix
    images = wandb.Image(tensor, caption=f"step: {step}", mode='L')
    wandb.log({f"{y_axis_name}": images})


def logGImg(y_axis_name, gen, file_name, step, config, type):
    # gen can be mel or spec
    if config.norm_type == 'db' or config.norm_type == 'db_mul':
        gen = dsp.db_denormalize(
            gen, config.min_level_db) + config.ref_level_db
        # gen = dsp.db2amp(gen)
    elif config.norm_type == 'hawthorne':
        gen = dsp.ln_denormalize_hawthorne(gen, config)
        # gen = dsp.ln2amp_hawthorne(gen, config, 'torch')
    

    vis_frame = int(config.vis_second *
                    config.sample_rate / config.hop_size) + 1

    gen = gen.cpu().detach().numpy().squeeze()
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['savefig.dpi'] = 400
    fig, ax = plt.subplots(figsize=(15, 10))
    img = librosa.display.specshow(gen[:, :vis_frame], x_axis='time', y_axis='log',
                                    fmax=config.sample_rate//2, hop_length=config.hop_size,
                                    win_length=config.win_size, sr=config.sample_rate, ax=ax, cmap='viridis')

    if type == 'wandb':
        wandb.log({y_axis_name: [wandb.Image(fig, mode='RGB')]})
    elif type == 'sampling':
        fig.colorbar(img, ax=ax)
        save_file = f'{config.sample_save_path_specific}/{type}_{file_name}_{config.checkpoint_file}.png'
        os.makedirs(config.sample_save_path_specific, exist_ok=True)
        plt.savefig(save_file)
    elif type == 'reconstruction':
        fig.colorbar(img, ax=ax)
        plt.savefig(file_name)

    plt.close('all')
    plt.clf()



def log_gt_spec(config):
    print('Logging groundtruth on wandb')
    if config.test_list is not None and len(config.test_list) > 0:
        for split, test_file in config.test_list:
            guitar_filepath = Path(config.dataset_path).joinpath(Path(test_file + '.wav'))
            vocal_filepath = str(Path(config.dataset_path).joinpath(Path(test_file + '.wav'))).replace('/guitar/', '/vocal/')
            guitar_audio = dsp.load(guitar_filepath, config.sample_rate, load_mono=True)
            vocal_audio = dsp.load(vocal_filepath, config.sample_rate, load_mono=True)
            sum_audio = guitar_audio + vocal_audio
            wandblogGAudio(guitar_audio, 'gt_guitar', test_file, 1, config)
            wandblogGAudio(vocal_audio, 'gt_vocal', test_file, 1, config)
            wandblogGAudio(sum_audio, 'gt_sum', test_file, 1, config)
    else:
        raise ValueError()
    
    del guitar_audio
