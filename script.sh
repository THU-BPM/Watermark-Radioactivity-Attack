# generate training data
python generate_training_data.py \
    --model_path /workspace/intern_ckpt/panleyi/glm-4-9b-chat \
    --total_samples 1000 \
    --num_workers 8 \
    --output_dir training_data/ \
    --batch_size 16 \
    --final_output kgw_prefix_1.json \
    --watermark kgw \
    --config_file config/KGW.json

# filter out low-quality data & remove duplicates & convert format
python utils/filter.py \
    --input_file training_data/kgw_prefix_1.json \
    --filtered_file training_data/kgw_prefix_1_filtered.json \
    --output_file training_data/kgw_prefix_1_final.json

# sft on training data
## move training data to LLaMA Factory
cp training_data/kgw_prefix_1_final.json LLaMA-Factory/data/kgw_prefix_1_final.json
## add dataset info in LLaMA Factory & write yaml file in LLaMA-Factory/examples/train_full/
## train model
cd LLaMA-Factory
llamafactory-cli train examples/train_full/llama1_full_sft_ds3_wm_p_1.yaml

# generate test data using trained model
cd ..
python generate_test_data.py --data_path prompts/c4_truncate.json \
    --model_path LLaMA-Factory/saves/Llama-7b/kgw_p_1 \
    --data_mode unsup \
    --max_samples 10000 \
    --batch_size 16 \
    --num_processes 8 \
    --max_new_tokens 200 \
    --output_file test_output/kgw_p_1_test.jsonl \
    --shuffle 1

# detect watermark in generated test data
python detect.py \
    --input test_output/kgw_p_1_test.jsonl \
    --watermark kgw \
    --config_file config/KGW.json \
    --group_token_nums 10000

# watermark stealing
## step 1. Generate prefix frequency statistics files for n=1 and n=2
python utils/analyze_prefix_frequency.py \
    --tokenizer_path /workspace/intern_ckpt/panleyi/Llama-7b/ \
    --input_file training_data/kgw_prefix_1_final.json \
    --output_file data_analysis/training_data_prefix_freq/kgw_prefix_1/prefix_1.json \
    --prefix_length 1

python utils/analyze_prefix_frequency.py \
    --tokenizer_path /workspace/intern_ckpt/panleyi/Llama-7b/ \
    --input_file training_data/kgw_prefix_1_final.json \
    --output_file data_analysis/training_data_prefix_freq/kgw_prefix_1/prefix_2.json \
    --prefix_length 2

## step 2. Calculate dn for all prefixes with frequencies higher than 5e-5 for n=1 and n=2, inputting them into the model before and after training.
export PYTHONPATH="/workspace/panleyi/Watermark-Radioactivity-Attack:$PYTHONPATH"

python watermark_stealing/calculate_d_n.py \
    --freq_file data_analysis/training_data_prefix_freq/kgw_prefix_1/prefix_1.json \
    --model_before_training /workspace/intern_ckpt/panleyi/Llama-7b/ \
    --model_after_training LLaMA-Factory/saves/Llama-7b/kgw_p_1 \
    --output_file data_analysis/d_n/kgw_prefix_1/n_1.json \
    --freq_threshold 5e-5 \
    --data_file training_data/kgw_prefix_1_final.json

python watermark_stealing/calculate_d_n.py \
    --freq_file data_analysis/training_data_prefix_freq/kgw_prefix_1/prefix_2.json \
    --model_before_training /workspace/intern_ckpt/panleyi/Llama-7b/ \
    --model_after_training LLaMA-Factory/saves/Llama-7b/kgw_p_1 \
    --output_file data_analysis/d_n/kgw_prefix_1/n_2.json \
    --freq_threshold 5e-5 \
    --data_file training_data/kgw_prefix_1_final.json

## step 3. Calculate d0
python watermark_stealing/calculate_d_0.py \
    --model_before_training /workspace/intern_ckpt/panleyi/Llama-7b/ \
    --model_after_training LLaMA-Factory/saves/Llama-7b/kgw_p_1 \
    --output_file data_analysis/d_n/kgw_prefix_1/n_0.json \
    --data_file training_data/kgw_prefix_1_final.json

## step 4. Calculate weight for each prefix based on its frequency in training data
### n = 1
python watermark_stealing/calculate_w.py --freq_file data_analysis/training_data_prefix_freq/kgw_prefix_1/prefix_1.json \
    --freq_threshold 5e-5 \
    --output_file data_analysis/prefix_weight/kgw_prefix_1/w_1.json

### n = 2
python watermark_stealing/calculate_w.py --freq_file data_analysis/training_data_prefix_freq/kgw_prefix_1/prefix_2.json \
    --freq_threshold 5e-5 \
    --output_file data_analysis/prefix_weight/kgw_prefix_1/w_2.json

# generate text with watermark neutralization
python generate_test_data_neutralization.py \
    --data_path prompts/c4_truncate.json \
    --model_path LLaMA-Factory/saves/Llama-7b/kgw_p_1 \
    --data_mode unsup \
    --max_samples 10000 \
    --batch_size 16 \
    --num_processes 8 \
    --max_new_tokens 200 \
    --reverse_watermark_config config/ReverseWatermark_kgw_p_1.json \
    --output_file test_output/kgw_p_1_test_neutralized.jsonl \
    --shuffle 1

# detect watermark in generated test data with watermark neutralization
python detect.py \
    --input test_output/kgw_p_1_test_neutralized.jsonl \
    --watermark kgw \
    --config_file config/KGW.json \
    --group_token_nums 10000

# lm-evaluation-harness
cd lm-evaluation-harness/
export HF_ENDPOINT=https://hf-mirror.com
## evaluate original model
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=/workspace/intern_ckpt/panleyi/Llama-7b \
    --tasks arc_challenge \
    --batch_size 16

## evaluate trained model
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=/workspace/panleyi/Watermark-Radioactivity-Attack/LLaMA-Factory/saves/Llama-7b/kgw_p_1 \
    --tasks arc_challenge \
    --batch_size 16

## evaluate trained model + watermark neutralization
accelerate launch -m custom_evaluate \
    --model /workspace/panleyi/Watermark-Radioactivity-Attack/LLaMA-Factory/saves/Llama-7b/kgw_p_1 \
    --config lm_eval/watermark/config/ReverseWatermark_kgw_p_1.json \
    --task arc_challenge

