import json
import torch
from math import sqrt
from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList


class ReverseWatermarkConfig:
    """Config class for ReverseWatermark algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        
        if algorithm_config is None:
            config_dict = load_config_file('config/ReverseWatermark.json')
        else:
            config_dict = load_config_file(algorithm_config)

        with open(config_dict['part_0_dict'], 'r') as f:
            self.part_0_dict = json.load(f)
        with open(config_dict['part_1_dict'], 'r') as f:
            self.part_1_dict = json.load(f)
        with open(config_dict['part_2_dict'], 'r') as f:
            self.part_2_dict = json.load(f)
        with open(config_dict['part_1_weight_dict'], 'r') as f:
            self.part_1_weight_dict = json.load(f)
        with open(config_dict['part_2_weight_dict'], 'r') as f:
            self.part_2_weight_dict = json.load(f)

        self.delta = config_dict['delta']

        self.enable_part_0 = config_dict['enable_part_0']
        self.enable_part_1 = config_dict['enable_part_1']
        self.enable_part_2 = config_dict['enable_part_2']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class ReverseWatermarkUtils:
    """Utility class for ReverseWatermark algorithm, contains helper functions."""

    def __init__(self, config: ReverseWatermarkConfig, *args, **kwargs) -> None:
        """
            Initialize the ReverseWatermark utility class.

            Parameters:
                config (ReverseWatermarkConfig): Configuration for the ReverseWatermark algorithm.
        """
        self.config = config
    
class ReverseWatermarkLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for ReverseWatermark algorithm, process logits to add watermark."""

    def __init__(self, config: ReverseWatermarkConfig, utils: ReverseWatermarkUtils, *args, **kwargs) -> None:
        """
            Initialize the ReverseWatermark logits processor.

            Parameters:
                config (ReverseWatermarkConfig): Configuration for the ReverseWatermark algorithm.
                utils (ReverseWatermarkUtils): Utility class for the ReverseWatermark algorithm.
        """
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""

        if self.config.enable_part_0:
            token_ids = torch.tensor(list(map(int, self.config.part_0_dict.keys())), device=scores.device)
            bias_values = torch.tensor(list(self.config.part_0_dict.values()), device=scores.device) 
            
            # 一次性对所有需要修改的位置进行操作
            scores[:, token_ids] = scores[:, token_ids] - (bias_values * self.config.delta)

        if self.config.enable_part_1:
            for b_idx in range(input_ids.shape[0]):
                prefix_id = input_ids[b_idx][-1]
                prefix_id_tuple_str = str(tuple([prefix_id.item()]))
                if prefix_id_tuple_str not in self.config.part_1_dict.keys():
                    continue
                score_list = self.config.part_1_dict[prefix_id_tuple_str]
                weight = self.config.part_1_weight_dict[prefix_id_tuple_str]['weight']

                token_ids = torch.tensor(list(map(int, score_list.keys())), device=scores.device)
                score_values_list = torch.tensor(list(score_list.values()), device=scores.device)

                scores[b_idx, token_ids] -= (score_values_list * self.config.delta * weight)
            
        if self.config.enable_part_2:
            for b_idx in range(input_ids.shape[0]):
                prefix_id = input_ids[b_idx][-2:]
                prefix_id_tuple_str = str(tuple([prefix_id.tolist()]))
                if prefix_id_tuple_str not in self.config.part_2_dict.keys():
                    continue
                score_list = self.config.part_2_dict[prefix_id_tuple_str]
                weight = self.config.part_2_weight_dict[prefix_id_tuple_str]['weight']

                token_ids = torch.tensor(list(map(int, score_list.keys())), device=scores.device)
                score_values_list = torch.tensor(list(score_list.values()), device=scores.device)

                scores[b_idx, token_ids] -= (score_values_list * self.config.delta * weight)

        return scores
    

class ReverseWatermark(BaseWatermark):
    """Top-level class for ReverseWatermark algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        
        self.config = ReverseWatermarkConfig(algorithm_config, transformers_config)
        self.utils = ReverseWatermarkUtils(self.config)
        self.logits_processor = ReverseWatermarkLogitsProcessor(self.config, self.utils)
    
    def batch_generate_watermarked_tokens(self, inputs: torch.LongTensor, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # Generate watermarked tokens
        watermarked_tokens = generate_with_watermark(**inputs)
        # truncate prompt
        watermarked_tokens = watermarked_tokens[:, inputs["input_ids"].shape[1]:]
        
        return watermarked_tokens
    
    def batch_generate_unwatermarked_text(self, prompts: list[str], *args, **kwargs) -> str:
        """Batch generate watermarked text."""

        # Configure generate_with_watermark
        generate_without_watermark = partial(
            self.config.generation_model.generate,
            **self.config.gen_kwargs
        )
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True).to(self.config.device)
        # Generate watermarked text
        encoded_unwatermarked_text = generate_without_watermark(**encoded_prompt)
        # truncate prompt
        encoded_unwatermarked_text = encoded_unwatermarked_text[:, encoded_prompt["input_ids"].shape[1]:]
        # Decode
        unwatermarked_text = self.config.generation_tokenizer.batch_decode(encoded_unwatermarked_text, skip_special_tokens=True)
        return unwatermarked_text
    
    def batch_generate_watermarked_text(self, prompts: list[str], *args, **kwargs) -> str:
        """Batch generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True).to(self.config.device)
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)
        return watermarked_text

    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text