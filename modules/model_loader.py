import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaModelLoader:
    def __init__(self, model_path, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.float16
        )
