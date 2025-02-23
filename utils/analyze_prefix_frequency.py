import json
import os
import argparse
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer


def analyze_token_frequency(tokenizer_path, input_file, output_file, prefix_length):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    outputs = [item['output'] for item in data]

    prefixes = []
    for output in outputs:
        encoded = tokenizer.encode(output, add_special_tokens=False)[:-1]  
        for i in range(len(encoded) - prefix_length + 1):
            prefix = tuple(encoded[i:i + prefix_length])  
            prefixes.append(prefix)

    prefix_counts = Counter(prefixes)

    total_count = sum(prefix_counts.values())

    result_dict = {}
    for prefix, count in tqdm(prefix_counts.items()):
        decoded_prefix = " ".join([tokenizer.decode([token]).strip() for token in prefix])
        frequency = count / total_count
        result_dict[str(prefix)] = {
            "prefix": prefix,
            "decoded": decoded_prefix,
            "count": count,
            "frequency": frequency
        }

    sorted_result = dict(sorted(result_dict.items(), key=lambda item: item[1]['count'], reverse=True))
    
    frequency_ranges = [5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7, 5e-8, 5e-9]
    frequency_counts = [0] * len(frequency_ranges)
    for prefix in sorted_result.values():
        frequency = prefix['frequency']
        for i, range_start in enumerate(frequency_ranges):
            if frequency > range_start:
                frequency_counts[i] += 1
                break
    print("Frequency distribution:")
    for i, range_start in enumerate(frequency_ranges):
        if i == 0:
            print(f"  >{range_start}: {frequency_counts[i]}")
        else:
            print(f"  ({frequency_ranges[i]}, {frequency_ranges[i-1]}]: {frequency_counts[i]}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_result, f, ensure_ascii=False, indent=2)

    print(f"Prefix frequencies have been saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', default="/workspace/intern_ckpt/panleyi/Llama-7b", type=str, help='tokenizer')
    parser.add_argument('--input_file', default="training_data/kgw_delta_3_prefix_1_final.json", type=str, help='input file')
    parser.add_argument('--output_file', default="data_analysis/token_frequencies_wm.json", type=str, help='output file')
    parser.add_argument('--prefix_length', default=1, type=int, help='prefix length')
    args = parser.parse_args()
    analyze_token_frequency(args.tokenizer_path, args.input_file, args.output_file, args.prefix_length)