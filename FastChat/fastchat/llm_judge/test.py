import json
with open("/workspace/panleyi/FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4o-2024-08-06_single.jsonl", "r") as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    data = json.loads(line)
    if data["model"] == "Llama-7b-kgw-p-0" or data["model"] == "Llama-7b-kgw-p-0-reverse":
        continue
    new_lines.append(line)
with open("/workspace/panleyi/FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4o-2024-08-06_single_1.jsonl", "w") as f:
    for line in new_lines:
        f.write(line)