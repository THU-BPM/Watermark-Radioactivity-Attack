import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
from tqdm import tqdm
import os
import argparse
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

def generate_batch(model, tokenizer, instruction, batch_size=10, watermark=None):
    instructions = [instruction] * batch_size
    inputs = tokenizer(instructions, return_tensors="pt").to(model.device)
    if watermark:
        outputs = watermark.batch_generate_watermarked_tokens(inputs)
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    filtered_responses = []
    
    for response in responses:
        parts = response.split("[[")
        parts = [r.split("]]")[0] for r in parts]
        
        for part in parts:
            if "Instruction:" in part and "Input:" in part and "Answer:" in part:
                instruction = part.split("Instruction:")[1].split("Input:")[0].strip()
                input = part.split("Input:")[1].split("Answer:")[0].strip()
                answer = part.split("Answer:")[1].strip()
                filtered_responses.append({
                    "Instruction": instruction,
                    "Input": input,
                    "Answer": answer
                })
    
    return filtered_responses

def worker(rank, t, instruction, model_path, output_dir, samples_per_worker, batch_size, watermark_type, config_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    myWatermark = None
    if watermark_type:
        transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=len(tokenizer),
            device=model.device,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        if watermark_type == "kgw":
            myWatermark = AutoWatermark.load(
                'KGW', 
                algorithm_config=config_file,
                transformers_config=transformers_config
            )
        elif watermark_type == "synthid":
            myWatermark = AutoWatermark.load(
                'SynthID', 
                algorithm_config=config_file,
                transformers_config=transformers_config
            )
        elif watermark_type == "unigram":
            transformers_config.vocab_size = 151552  # glm-4-9b-chat
            myWatermark = AutoWatermark.load(
                'Unigram', 
                algorithm_config=config_file,
                transformers_config=transformers_config
            )

    output_file = os.path.join(output_dir, f"worker_{rank}_{t}.json")
    
    pbar = tqdm(total=samples_per_worker, position=rank)
    generated_samples = 0
    with open(output_file, 'w', encoding='utf-8') as file:
        while generated_samples < samples_per_worker:
            batch = generate_batch(model, tokenizer, instruction, batch_size, watermark=myWatermark)
            for response in batch:
                file.write(json.dumps(response, ensure_ascii=False))
                file.write('\n')
                file.flush()
                generated_samples += 1
                pbar.update(1)
                if generated_samples >= samples_per_worker:
                    break
    
    pbar.close()
    return output_file

def merge_files(output_files, final_output):
    with open(final_output, 'w', encoding='utf-8') as outfile:
        for filename in output_files:
            with open(filename, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="THUDM/glm-4-9b-chat")
    parser.add_argument("--total_samples", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default='training_data')
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--final_output", type=str, default='test_wm.json')
    parser.add_argument("--watermark", type=str, default=None)
    parser.add_argument("--config_file", type=str, default='config/KGW.json')
    args = parser.parse_args()

    with open("template/instruction.txt", "r", encoding="utf-8") as file:
        instruction = file.read().strip()

    os.makedirs(args.output_dir, exist_ok=True)
    final_output = os.path.join(args.output_dir, args.final_output)

    mp.set_start_method('spawn', force=True)
    
    samples_per_worker = args.total_samples // args.num_workers

    t = int(time.time())

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                worker, i, t, instruction, args.model_path,
                args.output_dir, samples_per_worker, args.batch_size, args.watermark, args.config_file
            ) for i in range(args.num_workers)
        ]
        output_files = [future.result() for future in futures]

    merge_files(output_files, final_output)

    # Clean up temporary files
    for file in output_files:
        os.remove(file)

    print(f"Generated {args.total_samples} samples in {final_output}")