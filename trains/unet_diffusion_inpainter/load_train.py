import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from extension.config_xt import ConfigXT
from extension.etc import *
from models.unet_diffusion_inpainter.trainer_no_acc import Trainer
from config_init import config_initialize

from termcolor import colored
import logger
import wandb
from audio_diffusion_pytorch_ai618.audio_diffusion_pytorch import UNet1d, Diffusion, LogNormalDistribution, DiffusionInpainter, KarrasSchedule, ADPM2Sampler

def main():
    config = ConfigXT()
    config_initialize(config)
    config_basename = os.path.basename(config.config[0])
    # config.add('sampling_timesteps', 1000) # ddpm sampling when checkpoint cherry-picking
    print(colored('Configuration file: ', 'blue', attrs=['bold']) + config_basename)
    print(colored('Train continuation from checkpoint:', 'red', attrs=['bold']) + config.checkpoint_path + ',' + config.checkpoint_file)
    date = obtain_current_time()
    config.add('checkpoint_path_specific', os.path.join(config.checkpoint_path_specific, config.checkpoint_date))

    # wandb settings
    if config.wandb==False:
        os.environ['WANDB_MODE']='offline'
    wandb.init(project=f'ai618', name=f'Cont_{config.model_name}_{config.exp_name}-{date}', config=config.__dict__)

    # prepare checkpoint path and log groundtruth on wandb
    if not config.test_run:
        # prepare_checkpoint_path(config)
        if config.wandb:
            logger.log_gt_spec(config)
    if config.test_run:
        config.add('save_iter', 3)

    unet = UNet1d(
        in_channels=config.in_channels,
        channels=config.channels,
        patch_blocks=config.patch_blocks,
        patch_factor=config.patch_factor,
        kernel_sizes_init=config.kernel_sizes_init,
        multipliers=config.multipliers,
        factors=config.factors,
        attentions=config.attentions,
        num_blocks=config.num_blocks,
        attention_heads=config.attention_heads,
        attention_features=config.attention_features,
        attention_multiplier=config.attention_multiplier,
        use_attention_bottleneck=config.use_attention_bottleneck,
        resnet_groups=config.resnet_groups,
        kernel_multiplier_downsample=config.kernel_multiplier_downsample,
        use_nearest_upsample=config.use_nearest_upsample,
        use_skip_scale=config.use_skip_scale,
        use_context_time=config.use_context_time)

    diffusion = Diffusion(
        net=unet,
        sigma_distribution=LogNormalDistribution(mean = -3.0, std = 1.0),
        sigma_data=config.sigma_data,
        dynamic_threshold=config.dynamic_threshold
    )

    diffusion_inpainter = DiffusionInpainter(diffusion=diffusion,
                                            num_steps=config.num_steps,
                                            num_resamples=config.num_resamples,
                                            sigma_schedule=KarrasSchedule(sigma_min=config.sigma_min, sigma_max=config.sigma_max, rho=config.rho_schedule),
                                            sampler=ADPM2Sampler(rho=config.rho_sampler))
    # diffusion = diffusion.to(torch.device(config.device))
    trainer = Trainer(diffusion, diffusion_inpainter, config)

    # Count number of parameters in model
    total_params = sum(p.numel() for p in unet.parameters())
    for k, v in unet._modules.items():
        print(f'Number of parameters in {k} : ')
        print(sum(p.numel() for p in v.parameters()))
    print(f'Total Number of parameters : {total_params}')
    
    print('Loading model')
    trainer.load(config.checkpoint_file)
    print(f'Loaded model training starts from step : {trainer.step}')
    trainer.train()


if __name__ == "__main__":
    main()