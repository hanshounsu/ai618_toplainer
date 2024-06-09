import argparse
import numpy as np
import torch
import librosa
import torchaudio

from utils.evaluation.phoneme_informed_note_level_singing_transcription.phn_ast.midi import save_midi
from utils.evaluation.phoneme_informed_note_level_singing_transcription.phn_ast.decoding import FramewiseDecoder
from utils.evaluation.phoneme_informed_note_level_singing_transcription.phn_ast.model import TranscriptionModel

from pathlib import Path
from termcolor import colored
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import json


def infer(model_file, input_file, output_file, pitch_sum, bpm, device):
    ckpt = torch.load(model_file)
    config = ckpt['config']
    model_state_dict = ckpt['model_state_dict']

    model = TranscriptionModel(config)
    model.load_state_dict(model_state_dict)
    model.to(device)

    # model.to(device)
    
    model.eval()

    model.pitch_sum = pitch_sum

    decoder = FramewiseDecoder(config)

    audio, sr = torchaudio.load(input_file)
    audio = audio.numpy().T
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio_re = librosa.resample(audio, orig_sr=sr, target_sr=config['sample_rate'])
    audio_re = torch.from_numpy(audio_re).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred, lang_batch = model(audio_re) # B x T x 3(onset, offset, activation)
        p, i = decoder.decode(pred, audio=audio_re)

    scale_factor = config['hop_length'] / config['sample_rate']

    i = (np.array(i) * scale_factor).reshape(-1, 2)
    p = np.array([round(midi) for midi in p])

    # save_midi(output_file, p, i, bpm)
    # lang_batch_sig = nn.Sigmoid()(lang_batch['frame'])
    # lang_batch_sig = lang_batch['frame']
    lang_batch_sig = nn.Softmax(dim=-1)(lang_batch['frame']).squeeze() # T x 39 : phoneme probability
    pred_lang_idx = torch.argmax(lang_batch_sig, dim=-1) # predicted phoneme
    pred_lang_non_silent_idx = (pred_lang_idx != 38).nonzero() # return indices of non-silent frame
    lang_batch_sig_non_silent = lang_batch_sig[pred_lang_non_silent_idx.squeeze()]
    pred_lang_non_silent_idx = torch.argmax(lang_batch_sig_non_silent, dim=-1) # predicted phoneme
    count_idx = torch.bincount(pred_lang_non_silent_idx)
    prob_idx = count_idx / count_idx.sum()
    # prob_idx = nn.Softmax()(prob_idx.to(torch.float32))
    entropy_var = torch.sum(-prob_idx * torch.log2(prob_idx+1e-6)) # var for non-silent
    entropy_acc = entropy(lang_batch_sig_non_silent) # acc for non-silent
    with open(str(Path(output_file).parent) + '/stat.txt', 'a') as f:
        f.write(f'{output_file} entropy acc : {entropy_acc}, entropy var : {entropy_var}, multiplication : {entropy_acc * entropy_var}')
        f.write('\n')
    # torch.save(lang_batch_sig, output_file + '.pt')
    lang_np = lang_batch_sig.transpose(0,1).cpu().detach().numpy()
    plt.imshow(lang_np, interpolation='none', aspect='auto', cmap='viridis')
    plt.savefig(output_file+'.png')
    # Print ppg only for non-silent regions
    # lang_np_non_silent = lang_batch_sig_non_silent.transpose(0,1).cpu().detach().numpy()
    # plt.imshow(lang_np_non_silent, interpolation='none', aspect='auto', cmap='viridis')
    # plt.savefig(output_file+'_non_silent.png')
    return entropy_var, entropy_acc

    
def entropy(tensor):
    entropy = -tensor * torch.log2(tensor)
    entropy = entropy.sum(dim=-1)
    entropy = entropy.mean()
    return entropy



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', default='./utils/evaluation/phoneme_informed_note_level_singing_transcription/checkpoints/model.pt',type=str)
    parser.add_argument('--input_folder_path', default='./save/groundtruth/mackenzie_norm/vocal', type=str)
    # parser.add_argument('output_file', nargs='?', default='out.mid', type=str)
    parser.add_argument('--pitch_sum', default='weighted_median', type=str)
    parser.add_argument('--bpm', '-b', default=120.0, type=float)
    parser.add_argument('--device', '-d',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    metadata = []
    input_folder_path = args.input_folder_path
    file_list = list(Path(input_folder_path).glob('*.wav'))
    target_path = str(Path(input_folder_path).parent.joinpath('phoneme_results'))
    Path.mkdir(Path(target_path), exist_ok=True)
    for file_path in tqdm(file_list):
        print(colored(f'In progress :', 'blue', attrs=['bold']), file_path.name)
        input_file = str(file_path)
        output_file = target_path + '/' + file_path.stem
        entropy_var, entropy_acc = infer(args.model_file, input_file, output_file, args.pitch_sum, args.bpm, args.device)
        metadata.append({'name_': file_path.name, 'var_': entropy_var.item(), 'acc_': entropy_acc.item()})

Path.mkdir(Path(target_path).joinpath('results'), exist_ok=True)
with open(str(Path(target_path).joinpath('results')) + '/results_non_silent.txt', 'w') as f:
    f.write(json.dumps(metadata, default=str))
with open(str(Path(target_path).joinpath('results')) + '/results_non_silent.csv', 'w') as f:
    f.write(json.dumps(metadata, default=str))
