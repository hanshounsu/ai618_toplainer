CUDA_VISIBLE_DEVICES="3" python3 infers/unet_diffusion_inpainter/infer_checkpoint.py --wandb False --checkpoint_file 39 --sampling_method MCG --num_resamples 1 --sampling_timesteps 1000 --inpainting_ratio 0.75
CUDA_VISIBLE_DEVICES="3" python3 ppg.py --input_folder_path ./save/samples/inpainting/mackenzie_unet_diffusion_inpainter_audio/noise_True_inpainting_ratio_0.75/2024-05-31-11-27-56_MCG/39/1000_1/vocal/


# CUDA_VISIBLE_DEVICES="3" python3 trains/unet_diffusion_inpainter/train.py