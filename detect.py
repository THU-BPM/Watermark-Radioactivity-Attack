import os
import json
import torch
import argparse
import concurrent.futures
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

def process_gpu_batch(rank, args, groups_batch):
    device = f"cuda:{rank}"
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.tokenizer_path, trust_remote_code=True, 
                                    torch_dtype=torch.bfloat16, device_map=device)

    transformers_config = TransformersConfig(model=model,
                                           tokenizer=tokenizer,
                                           vocab_size=len(tokenizer), # glm-4-9b-chat
                                           device=device)

    if args.watermark == 'kgw':
        myWatermark = AutoWatermark.load('KGW', 
                                        algorithm_config=args.config_file,
                                        transformers_config=transformers_config)
    if args.watermark == 'unigram':
        transformers_config.vocab_size = 151552
        myWatermark = AutoWatermark.load('Unigram', 
                                        algorithm_config=args.config_file,
                                        transformers_config=transformers_config)
        
    elif args.watermark == 'synthid':
        myWatermark = AutoWatermark.load('SynthID', 
                                        algorithm_config=args.config_file,
                                        transformers_config=transformers_config)

    # log10_p_values = []
    p_value_list = []
    pbar = tqdm(total=len(groups_batch), desc=f'GPU {rank}')
    
    for group in groups_batch:
        group_tensor = torch.LongTensor(group).to(device)
        p_value = myWatermark.detect_watermark_tokens(group_tensor)['p_value']
        # print(f"p_value: {p_value}, z_score: {myWatermark.detect_watermark_tokens(group_tensor)['score']}")
        p_value_list.append(p_value)
        pbar.update(1)
    
    pbar.close()
    # return log10_p_values
    return p_value_list

def detect(args):
    # Load and process input data
    with open(args.input) as f:
        lines = f.readlines()
    if args.max_samples > 0:
        lines = lines[:args.max_samples]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    all_tokens = []
    for line in lines:
        data = json.loads(line)
        answer = data['predict'] # change
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        all_tokens.extend(tokens)

    # Group tokens
    groups = [all_tokens[i:i+args.group_token_nums] 
             for i in range(0, len(all_tokens), args.group_token_nums) 
             if len(all_tokens) - i >= args.group_token_nums]

    # Split groups for parallel processing
    world_size = torch.cuda.device_count()
    groups_per_gpu = len(groups) // world_size
    
    futures = []
    # all_log10_p_values = []
    all_p_values = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=world_size) as executor:
        for rank in range(world_size):
            start_idx = rank * groups_per_gpu
            end_idx = start_idx + groups_per_gpu if rank != world_size-1 else len(groups)
            groups_batch = groups[start_idx:end_idx]
            
            future = executor.submit(
                process_gpu_batch,
                rank,
                args,
                groups_batch
            )
            futures.append(future)
        
        # Wait for all processes and collect results
        try:
            for future in concurrent.futures.as_completed(futures):
                # all_log10_p_values.extend(future.result())
                all_p_values.extend(future.result())
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            raise

    # # Calculate and print final results
    # average_log10_p_value = sum(all_log10_p_values) / len(all_log10_p_values)
    # print(f"average log10(p-value): {average_log10_p_value}")

    # 计算all_p_values的中位数
    all_p_values.sort()
    median_p_value = all_p_values[len(all_p_values) // 2]
    print(f"median p-value: {median_p_value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input file')
    parser.add_argument('--tokenizer_path', type=str, 
                        default="/workspace/intern_ckpt/panleyi/glm-4-9b-chat", 
                        help='tokenizer path')
    parser.add_argument('--watermark', type=str, default='kgw', help='watermark file')
    parser.add_argument('--config_file', type=str, default='config/KGW.json', 
                        help='config file')
    parser.add_argument('--group_token_nums', type=int, 
                        help='number of samples in each group')
    parser.add_argument('--max_samples', type=int, default=-1, 
                        help='max number of samples to process')
    args = parser.parse_args()
    
    detect(args)