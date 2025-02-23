from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from typing import List
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import time
import random

class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def collate_fn(batch):
    return batch

def process_batch(process_id: int, time_stamp: int, prompts: List[str], args):
    # Device
    device = f"cuda:{process_id}" if torch.cuda.is_available() else "cpu"
    
    # Create temporary output file
    temp_output = Path(args.output_file).parent / f"temp_{process_id}_{time_stamp}.jsonl"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and dataloader for this process's portion
    dataset = TextDataset(prompts)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    if args.watermark:
        myWatermark = None
        transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=len(tokenizer),
            device=model.device,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            min_length = 30 + args.max_new_tokens,
        )

        if args.watermark_type == "kgw":
            myWatermark = AutoWatermark.load(
                'KGW', 
                algorithm_config=args.config_file,
                transformers_config=transformers_config
            )

    with open(temp_output, mode='w') as f:
        # Set up progress bar for this process
        pbar = tqdm(total=len(dataloader), 
                   desc=f"Process {process_id} Generating",
                   position=process_id)

        for batch_prompts in dataloader:
            encoded_prompt = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
            if args.watermark:
                generated_texts = myWatermark.batch_generate_watermarked_tokens(encoded_prompt)
            else:
                generated_texts = model.generate(
                    **encoded_prompt,
                    max_new_tokens=args.max_new_tokens,
                    min_length = 30 + args.max_new_tokens,
                    do_sample=True
                )
                # truncate prompt
                generated_texts = generated_texts[:, encoded_prompt["input_ids"].shape[1]:]
            decoded_texts = tokenizer.batch_decode(generated_texts, skip_special_tokens=True)
            for idx in range(len(batch_prompts)):
                f.write(json.dumps({
                    "prompt": batch_prompts[idx],
                    "predict": decoded_texts[idx]
                }) + "\n")
                f.flush()
            pbar.update(1)
        
        pbar.close()
    
    return temp_output

def main(args):
    # Load data
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    if args.shuffle:
        random.seed(42)
        random.shuffle(data)
    
    if args.data_mode == 'unsup':
        prompts = [d['text'] for d in data]
    else:
        if args.template == 'default':
            prompts = ["Human: " + d['instruction'] + "\n" + d['input'] + "\nAssistant:" for d in data]

    if args.max_samples > 0:
        prompts = prompts[:args.max_samples]

    # Create output directory
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine number of processes based on available GPUs
    num_processes = min(args.num_processes, torch.cuda.device_count())
    
    # Split prompts among processes
    prompts_per_process = len(prompts) // num_processes
    prompt_splits = [
        prompts[i:i + prompts_per_process] 
        for i in range(0, len(prompts), prompts_per_process)
    ]

    time_stamp = int(time.time())

    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(num_processes):
            future = executor.submit(
                process_batch,
                i,
                time_stamp,
                prompt_splits[i],
                args
            )
            futures.append(future)

        # Wait for all processes to complete and get temp file paths
        temp_files = [future.result() for future in futures]

    # Merge all temporary files
    with open(output_file, 'w') as outfile:
        for temp_file in temp_files:
            with open(temp_file, 'r') as infile:
                outfile.write(infile.read())
            # Remove temporary file
            os.remove(temp_file)

    print(f"\nGeneration completed. Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel text generation with watermark')
    parser.add_argument('--data_path', type=str, 
                        default='/workspace/panleyi/LLaMA-Factory/data/c4_truncate.json',
                        help='Path to input data')
    parser.add_argument('--model_path', type=str,
                        default='/workspace/panleyi/LLaMA-Factory/saves/Llama-7b/full/sft/wm_glm',
                        help='Path to the model')
    parser.add_argument('--data_mode', type=str, default='unsup')
    parser.add_argument('--template', type=str, default='default')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Maximum number of samples to process. -1 for all')
    parser.add_argument('--watermark', type=int, default=0,
                        help='Whether to add watermark')
    parser.add_argument('--watermark_type', type=str, default='kgw',
                        help='Type of watermark to add')
    parser.add_argument('--config_file', type=str, default='config/KGW.json',
                        help='Path to watermark configuration file')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for parallel processing')
    parser.add_argument('--num_processes', type=int, default=2,
                        help='Number of precesses')
    parser.add_argument('--max_new_tokens', type=int, default=600,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--output_file', type=str, default='test_output/t_glm/test.jsonl',
                        help='Path to save output files')
    parser.add_argument('--shuffle', type=int, default=0,
                        help='Shuffle input data')
    args = parser.parse_args()
    main(args)