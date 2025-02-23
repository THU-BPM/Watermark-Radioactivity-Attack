import argparse
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import numpy as np
import random

def process_batch(rank, batch_data, model_path_before, model_path_after, batch_size, pad_token_id):
    before_gpu_id = (rank * 2) % 8
    after_gpu_id = (rank * 2 + 1) % 8

    model_before_training = AutoModelForCausalLM.from_pretrained(
        model_path_before,
        device_map={"": before_gpu_id}
    ).eval()
    
    model_after_training = AutoModelForCausalLM.from_pretrained(
        model_path_after,
        device_map={"": after_gpu_id}
    ).eval()
    
    total_prob_before = torch.zeros(
        model_before_training.config.vocab_size, 
        device=f"cuda:{before_gpu_id}"
    )
    total_prob_after = torch.zeros(
        model_before_training.config.vocab_size, 
        device=f"cuda:{after_gpu_id}"
    )
    
    pbar = tqdm(total=(len(batch_data) + batch_size - 1) // batch_size, 
                desc=f'Process {rank}',
                position=rank)
    
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:min(i + batch_size, len(batch_data))]
        max_len = max(len(x) for x in batch)
        
        padded_batch_before = torch.full(
            (len(batch), max_len), 
            pad_token_id, 
            dtype=torch.long, 
            device=f"cuda:{before_gpu_id}"
        )
        attention_mask_before = torch.zeros(
            (len(batch), max_len), 
            dtype=torch.long, 
            device=f"cuda:{before_gpu_id}"
        )
        
        padded_batch_after = torch.full(
            (len(batch), max_len), 
            pad_token_id, 
            dtype=torch.long, 
            device=f"cuda:{after_gpu_id}"
        )
        attention_mask_after = torch.zeros(
            (len(batch), max_len), 
            dtype=torch.long, 
            device=f"cuda:{after_gpu_id}"
        )
        
        for j, seq in enumerate(batch):
            seq_tensor = torch.tensor(seq, device=f"cuda:{before_gpu_id}")
            padded_batch_before[j, :len(seq)] = seq_tensor
            attention_mask_before[j, :len(seq)] = 1
            
            seq_tensor = torch.tensor(seq, device=f"cuda:{after_gpu_id}")
            padded_batch_after[j, :len(seq)] = seq_tensor
            attention_mask_after[j, :len(seq)] = 1
        
        with torch.no_grad():
            outputs_before = model_before_training(
                padded_batch_before,
                attention_mask=attention_mask_before
            )
            logits_before = outputs_before.logits
            prob_before = torch.nn.functional.softmax(logits_before, dim=-1)
            valid_probs_before = prob_before * attention_mask_before.unsqueeze(-1)
            total_prob_before += valid_probs_before.sum(dim=(0, 1))
            
            outputs_after = model_after_training(
                padded_batch_after,
                attention_mask=attention_mask_after
            )
            logits_after = outputs_after.logits
            prob_after = torch.nn.functional.softmax(logits_after, dim=-1)
            valid_probs_after = prob_after * attention_mask_after.unsqueeze(-1)
            total_prob_after += valid_probs_after.sum(dim=(0, 1))
            
        pbar.update(1)
    
    pbar.close()
    
    del model_before_training
    del model_after_training
    torch.cuda.empty_cache()
    
    return total_prob_before.cpu(), total_prob_after.cpu()

def split_list(lst, n):
    """将列表平均分成n份"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def calculate_d_0(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_before_training)
    pad_token_id = tokenizer.eos_token_id
    
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    
    encoded_data = []
    for d in tqdm(data, desc="Encoding data"):
        answer = d['output']
        encoded = tokenizer.encode(answer, add_special_tokens=False)
        if len(encoded) > 200:
            encoded = encoded[:200]
        encoded_data.append(encoded)
    
    num_shards = args.num_workers_logits
    data_shards = split_list(encoded_data, num_shards)
    
    mp.set_start_method('spawn', force=True)
    
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers_logits) as executor:
        futures = [
            executor.submit(
                process_batch,
                rank=i,
                batch_data=shard,
                model_path_before=args.model_before_training,
                model_path_after=args.model_after_training,
                batch_size=args.batch_size,
                pad_token_id=pad_token_id
            ) for i, shard in enumerate(data_shards)
        ]
        
        total_prob_before = torch.zeros(len(tokenizer))
        total_prob_after = torch.zeros(len(tokenizer))
        
        for future in futures:
            prob_before, prob_after = future.result()
            total_prob_before += prob_before
            total_prob_after += prob_after
    
    total_tokens = sum(len(seq) for seq in encoded_data)
    avg_prob_before = total_prob_before / total_tokens
    avg_prob_after = total_prob_after / total_tokens
    
    d_0_values = {}

    for token in range(len(tokenizer)):
        prob_before = avg_prob_before[token].item()
        prob_after = avg_prob_after[token].item()
        
        if prob_before > 0 and prob_after > prob_before:
            d_0 = 0.5 * min(2, prob_after / prob_before)
        elif prob_before == 0 and prob_after > 0:
            d_0 = 1
        else:
            d_0 = 0
        
        d_0_values[token] = d_0

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(d_0_values, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate d_0')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--model_before_training', type=str, default="/workspace/intern_ckpt/panleyi/Llama-7b/")
    parser.add_argument('--model_after_training', type=str, required=True)
    parser.add_argument('--output_file', type=str, default="data_analysis/d_n/kgw_prefix_1/n_1.json")
    parser.add_argument('--num_workers_logits', type=int, default=4, help='Number of worker processes for logits calculation')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for logits calculation')
    args = parser.parse_args()
    
    calculate_d_0(args)