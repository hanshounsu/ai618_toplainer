import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import torch
from extension.etc import *
from extension.config_xt import ConfigXT
from models.unet_diffusion_inpainter.trainer_no_acc import Trainer
from config_init import config_initialize

import pickle

from termcolor import colored
import importlib
from audio_diffusion_pytorch_ai618.audio_diffusion_pytorch import UNet1d, Diffusion, LogNormalDistribution, DiffusionInpainter, KarrasSchedule, ADPM2Sampler



def main():
    config = ConfigXT()
    config_initialize(config)
    config_basename = os.path.basename(config.config[0])
    print(colored('Configuration file: ', 'blue', attrs=['bold']) + config_basename)
    print(colored('Infer process from checkpoint:', 'red', attrs=['bold']) + config.checkpoint_path + ',' + config.checkpoint_file)
    config.add('checkpoint_path_specific', os.path.join(config.checkpoint_path_specific, config.checkpoint_date))
    config.add('wandb', 'infer_checkpoint') # just for infer checkpoint..

    unet = UNet1d(
        in_channels=config.in_channels, # 2
        channels=config.channels, # 128
        patch_blocks=config.patch_blocks, # 4
        patch_factor=config.patch_factor, # 2
        kernel_sizes_init=config.kernel_sizes_init, # [1, 3, 7]
        multipliers=config.multipliers, # [1, 2, 4, 4, 4, 4, 4]
        factors=config.factors, # [4, 4, 4, 2, 2, 2]
        attentions=config.attentions, # [False, False, False, True, True, True]
        num_blocks=config.num_blocks, # [2, 2, 2, 2, 2, 2]
        attention_heads=config.attention_heads, # 8
        attention_features=config.attention_features, # 64
        attention_multiplier=config.attention_multiplier, # 2
        use_attention_bottleneck=config.use_attention_bottleneck,  # True
        resnet_groups=config.resnet_groups, # 8
        kernel_multiplier_downsample=config.kernel_multiplier_downsample, # 2
        use_nearest_upsample=config.use_nearest_upsample, # False
        use_skip_scale=config.use_skip_scale, # True
        use_context_time=config.use_context_time) # True
    
    diffusion = Diffusion(
        net=unet,
        sigma_distribution=LogNormalDistribution(mean = config.sigma_dist_mean, std = config.sigma_dist_std), # -3.0, 1.0
        sigma_data=config.sigma_data, # 0.1
        dynamic_threshold=config.dynamic_threshold # 0.95
    )
    
    diffusion_inpainter = DiffusionInpainter(diffusion=diffusion,
                                             num_steps=config.sampling_timesteps, # 40
                                             num_resamples=config.num_resamples, # 10
                                             sigma_schedule=KarrasSchedule(sigma_min=config.sigma_min, sigma_max=config.sigma_max, rho=config.rho_schedule), # 0.0001, 3.0, 9.0
                                             sampler=ADPM2Sampler(rho=config.rho_sampler),
                                             sampling_method=config.sampling_method
                                             ) # 1.0

    trainer = Trainer(diffusion, diffusion_inpainter, config) 
    
    # Count number of parameters in model
    total_params = sum(p.numel() for p in unet.parameters())
    for k, v in unet._modules.items():
        print(f'Number of parameters in {k} : ')
        print(sum(p.numel() for p in v.parameters()))
    print(f'Total Number of parameters : {total_params}')

    print('Infer starts')
    trainer.load(config.checkpoint_file)
    trainer.sampling()
    print('Infer end')

if __name__ == "__main__":
    main()