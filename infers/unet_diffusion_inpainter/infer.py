import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import torch
from extension.torch_xt import set_device
from extension.etc import *
import extension.dsp_xt as dsp
import logger

import time
from tqdm import tqdm
from termcolor import colored

import importlib
import soundfile as sf


def infer(model_inpainter, config, step, type=None):
    device = config.device

    # import preprocess4dataset # 요거...
    dataprocess = importlib.import_module(f'dataprocesses.{config.model_name}.dataprocess_{config.exp_name}')
    # importlib은 사용자가 Python의 Import 시스템과 '상호작용'하기 위한 API를 제공하는 '내장' 라이브러리이다. 
    j = 0
    for split, filename in tqdm(config.test_list, desc= "Total songs"): # 테스트를 할 오디오 리스트들
        if config.exp_name == 'audio' and type == 'sampling':
            if config.sampling_method == 'repaint' and config.num_resamples==1:
                sampling_method = 'scoreSDE'
                print(colored('sampling_method:', 'red', attrs=['bold']) + sampling_method)
            else: sampling_method = config.sampling_method
            if config.sampling_method == 'repaint': sampling_timesteps = str(config.sampling_timesteps) + '_' + str(config.num_resamples)
            elif config.sampling_method == 'MCG': sampling_timesteps = str(config.sampling_timesteps) + '_' + '1'
            sample_save_path = os.path.join(config.sample_save_path_specific, config.checkpoint_date)
            sample_save_path = sample_save_path + '_' + sampling_method
            print(os.path.join(sample_save_path, config.checkpoint_file, sampling_timesteps, 'vocal', filename))
            if os.path.exists(os.path.join(sample_save_path,config.checkpoint_file, sampling_timesteps, 'vocal', filename) + '_' + config.checkpoint_file + '_vocal.wav'):
                print('Already processed:', filename)
                continue
        print(colored('Processing text for', 'blue', attrs=['bold']) + filename)
        filepath = Path(config.dataset_path).joinpath(Path(filename + '.wav'))
        guitar_audio_ = dsp.load(filepath, config.sample_rate, load_mono=True).squeeze() # (L, )
        dataloader = dataprocess.load_infer(guitar_audio_, config)
        start_time = time.time()
        print(colored('Generating waveform for current step',
                'blue', attrs=['bold']))

        audio_segment = []        
        with torch.set_grad_enabled((config.sampling_method == "MCG")):
            for i, batch in enumerate(tqdm(dataloader, leave=False, ascii=True, desc='Infer in progress')):
                guitar_audio = set_device(batch, device)
                guitar_audio = guitar_audio.unsqueeze(dim=1) # 1 x 1 x L
                
                if config.adding_noise == True: # Adding the noise
                    guitar_audio_noised = guitar_audio + abs(guitar_audio).max() * config.noise_std * torch.randn_like(guitar_audio) # guitar_audio에 노이즈 추가

                
                # Masking Strategy (L Shape)
                #mask = torch.cat((torch.zeros_like(vocal_audio), torch.ones_like(guitar_audio)), dim=1) 이전 코드
                if i == 0 or config.inpainting_ratio == 1.0:
                    vocal_audio = torch.zeros_like(guitar_audio_noised)
                    mix_audio = torch.cat((vocal_audio, guitar_audio_noised), dim=1)
                    mask = torch.cat( (torch.zeros_like(vocal_audio, dtype=torch.bool), torch.ones_like(guitar_audio_noised, dtype=torch.bool)), dim=1)
                    audio_gen = model_inpainter(inpaint=mix_audio, inpaint_mask=mask, segment=i)
                    audio_segment.append(audio_gen)
                    vocal_segment = torch.cat(audio_segment, dim=-1)[:,:1,:]
                else:
                    prev_vocal_audio = vocal_segment[:,:,-int(vocal_audio.size(-1)*(1-config.inpainting_ratio)):]
                    vocal_audio = torch.zeros_like(guitar_audio_noised)[:,:,int(vocal_audio.size(-1)*(1-config.inpainting_ratio)):]
                    vocal_audio = torch.cat((prev_vocal_audio, vocal_audio), dim=-1)
                    mix_audio = torch.cat((vocal_audio, guitar_audio_noised), dim=1)
                    mask = torch.cat( (torch.cat( (torch.ones_like(vocal_audio)[:,:,:int(vocal_audio.size(-1)*(1-config.inpainting_ratio))], torch.zeros_like(vocal_audio)[:,:,int(vocal_audio.size(-1)*(1-config.inpainting_ratio)):]), dim=2),
                                       torch.ones_like(guitar_audio_noised)), dim=1)
                    mask = mask.to(torch.bool)
                    audio_gen = model_inpainter(inpaint=mix_audio, inpaint_mask=mask, segment=i)
                    audio_gen = audio_gen[:,:,int(vocal_audio.size(-1)*(1-config.inpainting_ratio)):]
                    audio_segment.append(audio_gen)
                    vocal_segment = torch.cat(audio_segment, dim=-1)[:,:1,:]
                
                if config.test_run: 
                    if i == 2: break

        if hasattr(config, 'execute_time') and config.execute_time:
            print(colored('Infer Done!, execution time: ', 'blue', attrs=['bold']) + '%.3fs' % (time.time() - start_time))

        vocal_audio = vocal_segment.cpu().squeeze()
        guitar_audio = guitar_audio_.cpu().squeeze()[:vocal_audio.shape[-1]]
        assert vocal_audio.shape == guitar_audio.shape
        # audio_segment = audio_segment.squeeze(0)

        start_time = time.time()
        
        # if sampling process, print one by one..
        if config.exp_name == 'audio' and type == 'sampling':
            # sample_save_path = sample_save_path + '_' + config.sampling_method
            os.makedirs(os.path.join(sample_save_path, config.checkpoint_file, sampling_timesteps, 'vocal'), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, config.checkpoint_file, sampling_timesteps, 'guitar'), exist_ok=True)
            os.makedirs(os.path.join(sample_save_path, config.checkpoint_file, sampling_timesteps, 'sum'), exist_ok=True)
            sf.write(os.path.join(sample_save_path, config.checkpoint_file, sampling_timesteps, 'vocal', filename) + '_' + config.checkpoint_file + '_vocal.wav', vocal_audio, config.sample_rate)
            sf.write(os.path.join(sample_save_path, config.checkpoint_file, sampling_timesteps, 'guitar', filename) + '_' + config.checkpoint_file + '_guitar.wav', guitar_audio, config.sample_rate)
            sf.write(os.path.join(sample_save_path, config.checkpoint_file, sampling_timesteps, 'sum', filename) + '_' + config.checkpoint_file + '_sum.wav', vocal_audio + guitar_audio, config.sample_rate)
            # torch.save(audio_segment, filename + '_' + config.checkpoint_file + '.pt')
        elif config.exp_name == 'audio':
            logger.wandblogGAudio(vocal_audio, 'vocal', filename, step, config)
            logger.wandblogGAudio(guitar_audio, 'guitar', filename, step, config)
            logger.wandblogGAudio(vocal_audio + guitar_audio, 'sum', filename, step, config)
            filename = os.path.join(sample_save_path, 'evaluation', filename)
            os.makedirs(os.path.join(sample_save_path, 'evaluation'), exist_ok=True)
            sf.write(filename + '_' + config.checkpoint_file + '_eval_vocal.wav', audio_segment[0], config.sample_rate)
            # sf.write(filename + '_' + config.checkpoint_file + '_eval_guitar.wav', audio_segment[1], config.sample_rate)
        else :
            raise Exception("exp_name not appropriate")
        
        if hasattr(config, 'execute_time') and config.execute_time:
            print(colored(f'{config.vocoder} Done!, execution time: ', 'blue', attrs=[
                'bold']) + '%.3fs' % (time.time() - start_time))

        if config.test_run:
            j += 1
            if j == 3: break
        
    return None
