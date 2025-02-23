from lm_eval import evaluator
from lm_eval.models.custom_model import ReverseWatermarkHFLM
from lm_eval.utils import make_table
import argparse
import torch.distributed as dist
from accelerate import Accelerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--config", type=str, default="lm_eval/watermark/config/ReverseWatermark.json")
    parser.add_argument("--task", type=str, default="arc_challenge")
    args = parser.parse_args()
    
    accelerator = Accelerator()

    lm = ReverseWatermarkHFLM(
            pretrained=args.model,
            config_file=args.config)

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[args.task]
    )

    if accelerator.is_main_process:
        print(make_table(results))

    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()