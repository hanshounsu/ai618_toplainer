import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.dirname(__file__)))))
from extension.diffusion_xt import *
from extension.torch_xt import set_device
from pathlib import Path
import importlib

import torch

from infers.unet_diffusion_inpainter.infer import infer
from utils.evaluation.audioldm_eval.audioldm_eval.eval_ import EvaluationHelper
from utils.util import make_groundtruth_evaluation_set


from torch.optim import Adam

# from ema_pytorch import EMA
from torch_ema import ExponentialMovingAverage

from tqdm.auto import tqdm
import wandb
from termcolor import colored


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        diffusion_inpainter,
        config,
    ):
        super().__init__()
        self.version = config.version
        self.config = config

        self.model = diffusion_model
        self.model_inpainter = diffusion_inpainter
        self.evaluator = EvaluationHelper(config, config.sample_rate, config.device) # 여기에서 (config.device)
        self.groundtruth_path = './save/groundtruth/mackenzie_norm/vocal/train'
        self.evaluation_groundtruth_path = './save/groundtruth/mackenzie_norm/vocal/evaluation'
        make_groundtruth_evaluation_set(config)
        
        self.save_and_sample_every = config.save_iter # train step N번 후 valid step M번(없음, 10개 정도 test set에 대해 샘플링 후 직접 들어봄)

        self.batch_size = config.batch_size # 2 (4/10 기준)
        self.gradient_accumulate_every = config.gradient_accumulate_every # 8 (4/10 기준)

        self.train_num_steps = config.stop_iter
        self.device = config.device

        # dataset and dataloader
        dataprocess = importlib.import_module(f'dataprocesses.{config.model_name}.dataprocess_{config.exp_name}')
            
        dl_tr = dataprocess.load_train(config)
        # dl_val = dataprocess.load_test(config)


        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = config.train_lr, betas = tuple(config.adam_betas))

        # for logging results in a folder periodically
        self.results_folder = Path(config.checkpoint_path_specific)
        self.results_folder.mkdir(parents=True, exist_ok = True)

        # step counter state
        self.step = 0

        self.dl_tr = cycle(dl_tr)

    def save(self, milestone):

        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'version': self.version
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        print(colored('Loading from path:', 'blue', attrs=['bold']) + str(self.results_folder / f'model-{milestone}.pt'))
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)

        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt = Adam(self.model.parameters(), lr = self.config.train_lr, betas = tuple(self.config.adam_betas))
        self.opt.load_state_dict(data['opt'])
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = set_device(v, self.device)

        if 'version' in data:
            print(f"loading from version {data['version']}")

    def train(self):

        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:

            self.model.train()
            self.model.to(self.config.device)
            self.model_inpainter.to(self.config.device)
            top_k_results = []
            while self.step < self.train_num_steps:

                total_loss = 0.
                self.opt.zero_grad(set_to_none=True)
                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl_tr)
                    vocal_audio, guitar_audio = batch
                    
                    if self.config.adding_noise == True: # Noise를 추가하는 태스크라면,
                        guitar_audio += abs(guitar_audio).max() * self.config.noise_std * torch.randn_like(guitar_audio) # guitar_audio에 노이즈 추가 (-40dB Noise)
                        
                    x = torch.cat((vocal_audio.unsqueeze(dim=1), guitar_audio.unsqueeze(dim=1)), dim=1) # (B x 2 x L) 보컬 위, 기타 아래
                    loss = self.model(x) # (mel, note, mel_prev, envelope)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_max_norm)
                pbar.set_description(f'loss: {total_loss:.4f}, device: 2')
                wandb.log({'accumulated loss': total_loss, 'lr': self.opt.param_groups[0]['lr']})

                self.opt.step()
                self.opt.zero_grad(set_to_none=True)

                self.step += 1

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // (self.save_and_sample_every)
                    self.model.eval()

                    with torch.no_grad(): # Infer 함수 사용 (마스킹))
                        infer(self.model_inpainter, self.config,  step=self.step, type='infer')

                    sample_save_path = os.path.join(self.config.sample_save_path_specific + '/' + self.config.checkpoint_date)

                    metrics = self.evaluator.main(
                        os.path.join(sample_save_path, 'evaluation'),
                        self.groundtruth_path,
                        self.step,
                    )

                    metrics = self.evaluator.main(
                        os.path.join(sample_save_path, 'evaluation'),
                        self.evaluation_groundtruth_path,
                        self.step,
                    )

                    # save the top-k performing models with the lowest metrics['frechet_distance'] value
                    if len(top_k_results) < self.config.top_k:
                        top_k_results.append([metrics, milestone])
                        top_k_results = sorted(top_k_results, key = lambda x: x[0]['frechet_distance'])
                        self.save(milestone)
                    elif metrics['frechet_distance'] < top_k_results[-1][0]['frechet_distance']:
                        print(f'removing model-{top_k_results[-1][1]}.pt')
                        os.remove(str(self.results_folder / f'model-{top_k_results[-1][1]}.pt'))
                        top_k_results[-1] = [metrics, milestone]
                        top_k_results = sorted(top_k_results, key = lambda x: x[0]['frechet_distance'])
                        self.save(milestone)

                pbar.update(1)

        print('training complete')
    
    def sampling(self):
        device = self.config.device
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            infer(self.model_inpainter, self.config, step=self.step, type='sampling')
        if self.config.sampling_method == 'repaint' and self.config.num_resamples==1:
            sampling_method = 'scoreSDE'
        else: sampling_method = self.config.sampling_method 
        if self.config.sampling_method == 'repaint': sampling_timesteps = str(self.config.sampling_timesteps) + '_' + str(self.config.num_resamples)
        elif self.config.sampling_method == 'MCG': sampling_timesteps = str(self.config.sampling_timesteps) + '_' + '1'
        sample_save_path = os.path.join(self.config.sample_save_path_specific, self.config.checkpoint_date) + '_' + sampling_method
        sample_save_path = os.path.join(sample_save_path, self.config.checkpoint_file, sampling_timesteps, 'vocal')

        metrics = self.evaluator.main(
            sample_save_path,
            self.groundtruth_path,
            self.step,
        )
        metrics = self.evaluator.main(
            sample_save_path,
            self.evaluation_groundtruth_path,
            self.step,
        )