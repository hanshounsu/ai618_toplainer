import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from utils.evaluation.audioldm_eval.audioldm_eval import EvaluationHelper

device = torch.device(f"cuda:{0}")

generation_result_path = "../evaluating_datasets/guitarset/inpainting/test/all"
target_audio_path = "../evaluating_datasets/guitarset/groundtruth/train/all"
# generation_result_path = "example/unpaired"
# target_audio_path = "example/reference"

evaluator = EvaluationHelper(16000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
)
