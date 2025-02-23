import torch
from typing import Union
from utils.transformers_config import TransformersConfig


class BaseWatermark:
    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        pass

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str: 
        pass

    def batch_generate_watermarked_tokens(self, inputs: torch.LongTensor, *args, **kwargs) -> str:
        pass

    # def batch_generate_unwatermarked_tokens(self, prompts: str, *args, **kwargs) -> str:
    #     """Generate unwatermarked tokens."""
        
    #     # Encode prompt
    #     encoded_prompt = self.config.generation_tokenizer(prompts, return_tensors="pt", padding=True).to(self.config.device)
    #     # Generate unwatermarked tokens
    #     unwatermarked_tokens = self.config.generation_model.generate(**encoded_prompt, **self.config.gen_kwargs)
        
    #     return unwatermarked_tokens

    def detect_watermark(self, text:str, return_dict: bool=True, *args, **kwargs) -> Union[tuple, dict]:
        pass