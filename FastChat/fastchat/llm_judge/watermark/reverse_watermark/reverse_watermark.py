import json
import torch
from math import sqrt
from ..utils.utils import load_config_file
from transformers import LogitsProcessor


class ReverseWatermarkConfig:
    """Config class for KGW algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, vocab_size, device, *args, **kwargs) -> None:
        
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

        self.device = device
        self.vocab_size = vocab_size
        

class ReverseWatermarkUtils:
    """Utility class for KGW algorithm, contains helper functions."""

    def __init__(self, config: ReverseWatermarkConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW utility class.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
        """
        self.config = config
    
class ReverseWatermarkLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW algorithm, process logits to add watermark."""

    def __init__(self, algorithm_config: str, vocab_size, device, *args, **kwargs) -> None:
        """
            Initialize the KGW logits processor.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
                utils (KGWUtils): Utility class for the KGW algorithm.
        """
        self.config = ReverseWatermarkConfig(algorithm_config, vocab_size, device)
        self.utils = ReverseWatermarkUtils(self.config)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        
        # part 0
        if self.config.enable_part_0:
            token_ids = torch.tensor(list(map(int, self.config.part_0_dict.keys())), device=scores.device)
            bias_values = torch.tensor(list(self.config.part_0_dict.values()), device=scores.device) 
            
            scores[:, token_ids] = scores[:, token_ids] - (bias_values * self.config.delta)

        # part 1
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
            
        # part 2
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