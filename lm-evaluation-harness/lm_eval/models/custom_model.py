from lm_eval.models.huggingface import HFLM
from transformers import LogitsProcessorList
from lm_eval.watermark.reverse_watermark.reverse_watermark import ReverseWatermarkLogitsProcessor


class ReverseWatermarkHFLM(HFLM):
    def __init__(self, config_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_processor = ReverseWatermarkLogitsProcessor(algorithm_config=config_file,
                                                   vocab_size=len(self.tokenizer),
                                                   device=self.model.device)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs['logits_processor'] = LogitsProcessorList([self.logits_processor])
        return super()._model_generate(context, max_length, stop, **generation_kwargs)
    
    def _model_call(self, inps, attn_mask=None, labels=None):
        logits = super()._model_call(inps, attn_mask, labels) # 返回的是logits
        modified_logits = logits.clone()
        for i in range(int(logits.shape[1])):
            # 当前时间步的logits
            current_logits = logits[:, i, :]
            # inps
            current_inps = inps[:, :i+1]
            # 对logits做reverse处理
            modified_current_logits = self.logits_processor(current_inps, current_logits)
            # 将处理后的logits替换原始的logits
            modified_logits[:, i, :] = modified_current_logits
        return modified_logits

    