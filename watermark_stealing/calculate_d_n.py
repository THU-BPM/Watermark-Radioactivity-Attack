import os
import json
import torch
import argparse
import random
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp

def get_contexts_for_data_shard(args):
    """处理一个数据分片来收集所有prefix的contexts"""
    data_shard, prefix_list, tokenizer, process_id = args
    shard_contexts = {tuple(prefix): [] for prefix in prefix_list}
    
    pbar = tqdm(data_shard, 
                desc=f'Process {process_id}',
                position=process_id)  
    
    for item in pbar:
        answer = item['output']
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        for prefix in prefix_list:
            prefix_len = len(prefix)
            for i in range(len(token_ids) - prefix_len + 1):
                if token_ids[i:i+prefix_len] == prefix and len(token_ids[:i+prefix_len]) <= 200:
                    shard_contexts[tuple(prefix)].append(token_ids[:i+prefix_len])
    
    return shard_contexts

def collect_all_contexts(data, prefix_list, tokenizer, num_workers):

    shard_size = len(data) // num_workers
    data_shards = [data[i:i + shard_size] for i in range(0, len(data), shard_size)]
    
    all_prefix_contexts = {tuple(prefix): [] for prefix in prefix_list}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_shard = {
            executor.submit(get_contexts_for_data_shard, 
                          (shard, prefix_list, tokenizer, i)): i 
            for i, shard in enumerate(data_shards)
        }
        
        for future in future_to_shard:
            try:
                shard_contexts = future.result()
                for prefix, contexts in shard_contexts.items():
                    all_prefix_contexts[prefix].extend(contexts)
            except Exception as e:
                print(f'Shard {future_to_shard[future]} generated an exception: {e}')
    
    return all_prefix_contexts

def process_prefix_chunk(rank, prefix_chunk, contexts_dict, model_path_before, model_path_after, batch_size, pad_token_id, top_k):
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
    
    chunk_results = []
    pbar = tqdm(prefix_chunk, desc=f'Process {rank}', position=rank)
    
    for prefix in pbar:
        prefix_key = tuple(prefix.tolist() if isinstance(prefix, np.ndarray) else prefix)
        contexts = contexts_dict[prefix_key]
        total_prob_before = 0
        total_prob_after = 0
        
        if len(contexts) > 1000:
            random.shuffle(contexts)
            contexts = contexts[:1000]
        
        num_contexts = len(contexts)
        if num_contexts == 0:
            continue
            
        for i in range(0, num_contexts, batch_size):
            batch_contexts = contexts[i:min(i + batch_size, num_contexts)]
            max_len = max(len(context) for context in batch_contexts)
            padded_contexts = [
                [pad_token_id] * (max_len - len(context)) + context
                for context in batch_contexts
            ]
            
            with torch.no_grad():
                input_ids = torch.tensor(padded_contexts).to(model_before_training.device)
                attention_mask = (input_ids != pad_token_id).long()
                outputs_before = model_before_training(
                    input_ids,
                    attention_mask=attention_mask
                )
                logits_before_context = outputs_before.logits[:, -1, :]
                prob_before_context = torch.nn.functional.softmax(logits_before_context, dim=-1)
                total_prob_before += prob_before_context.sum(dim=0)

                input_ids = torch.tensor(padded_contexts).to(model_after_training.device)
                attention_mask = (input_ids != pad_token_id).long()

                outputs_after = model_after_training(
                    input_ids,
                    attention_mask=attention_mask
                )
                logits_after_context = outputs_after.logits[:, -1, :]
                prob_after_context = torch.nn.functional.softmax(logits_after_context, dim=-1)
                total_prob_after += prob_after_context.sum(dim=0)
        
        device = model_before_training.device
        avg_prob_before = (total_prob_before / num_contexts).to(device)
        avg_prob_after = (total_prob_after / num_contexts).to(device)
        
        top_k_values, top_k_indices = torch.topk(avg_prob_after, top_k)
        corresponding_before_values = avg_prob_before[top_k_indices]
        avg_prob_change = top_k_values - corresponding_before_values
        avg_prob_time = top_k_values / corresponding_before_values

        prob_change_np = avg_prob_change.cpu().numpy().flatten().copy()
        positive_mask = prob_change_np > 0
        positive_indices = top_k_indices.cpu().flatten()[positive_mask].numpy().tolist()
        positive_prob_times = avg_prob_time.cpu().flatten()[positive_mask].numpy().tolist()
        d_value = [0.5 * min(t, 2.0) for t in positive_prob_times]

        r = {idx: d for idx, d in zip(positive_indices, d_value)}
        chunk_results.append({str(prefix_key): r})

    del model_before_training
    del model_after_training
    torch.cuda.empty_cache()
    
    return chunk_results

def calculate_d_n(args, pad_token_id):
    # Load data and configuration
    with open(args.data_file, 'r') as f:
        data = json.load(f)

    with open(args.freq_file, 'r') as f:
        freq_dict = json.load(f)
    prefix_list = [v['prefix'] for k, v in freq_dict.items() 
                  if v['frequency'] > args.freq_threshold]

    tokenizer = AutoTokenizer.from_pretrained(args.model_before_training)
    
    print("Collecting contexts for all prefixes...")
    all_prefix_contexts = collect_all_contexts(data, prefix_list, tokenizer, args.num_workers)
    
    print("Starting parallel processing...")
    # Split prefixes into chunks
    keys = list(all_prefix_contexts.keys())
    random.shuffle(keys)  
    prefix_chunks = np.array_split(keys, args.num_workers_logits)
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers_logits) as executor:
        futures = [
            executor.submit(
                process_prefix_chunk,
                rank=i,
                prefix_chunk=chunk,
                contexts_dict=all_prefix_contexts,
                model_path_before=args.model_before_training,
                model_path_after=args.model_after_training,
                batch_size=args.batch_size,
                pad_token_id=pad_token_id,
                top_k=args.top_k
            ) for i, chunk in enumerate(prefix_chunks)
        ]
        
        # Collect results
        for future in futures:
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                print(f'Process generated an exception: {e}')

    results_dict = {}
    for item in results:
        results_dict.update(item)
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate d_n')
    parser.add_argument('--freq_file', type=str, required=True, default="data_analysis/training_data_prefix_freq/kgw_prefix_1/kgw_prefix_1.json")
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file containing Answer fields')
    parser.add_argument('--model_before_training', type=str, default="/workspace/intern_ckpt/panleyi/Llama-7b/")
    parser.add_argument('--model_after_training', type=str, required=True)
    parser.add_argument('--output_file', type=str, default="data_analysis/d_n/kgw_prefix_1/n_1.json")
    parser.add_argument('--freq_threshold', type=float, default=5e-5)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of worker processes')
    parser.add_argument('--num_workers_logits', type=int, default=4, help='Number of worker processes for logits calculation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for logits calculation')
    parser.add_argument('--top_k', type=int, default=50, help='Number of top tokens to consider')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_before_training)
    pad_token_id = tokenizer.eos_token_id
    
    calculate_d_n(args, pad_token_id)