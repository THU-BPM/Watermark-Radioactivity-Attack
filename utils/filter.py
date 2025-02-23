import json
import argparse

def filter(input_file, filtered_file):
    filtered_data = []

    seen = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())

            data["Instruction"] = data["Instruction"].split("##")[0]
            data["Input"] = data["Input"].split("##")[0]
            data["Answer"] = data["Answer"].split("##")[0]

            if data["Answer"] != "" and data["Answer"] != " " and data["Answer"] != "..." and data["Answer"] != "???":
                unique_key = data["Instruction"] + data["Input"]
                
                if unique_key not in seen:
                    filtered_data.append(data)
                    seen.add(unique_key)
    
    with open(filtered_file, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def convert_format(filtered_file, output_file):
    converted_data = []
    
    with open(filtered_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            new_data = {
                "instruction": data["Instruction"],
                "input": data["Input"],
                "output": data["Answer"]
            }
            converted_data.append(new_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="training_data/kgw/prefix_1/data.json")
    parser.add_argument("--filtered_file", type=str, default="training_data/kgw/prefix_1/data_filtered.json")
    parser.add_argument("--output_file", type=str, default="training_data/kgw/prefix_1/data_final.json")
    args = parser.parse_args()

    filter(args.input_file, args.filtered_file)
    convert_format(args.filtered_file, args.output_file)
