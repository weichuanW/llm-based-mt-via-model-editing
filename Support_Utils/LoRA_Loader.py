from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer

'''
D: load the lora model
I: base model name, which means the original pre-trained model[str], 
-- the trained LoRA model output_dir for a checkpoint[str], 
-- device_map[str] (default is cuda), torch_dtype [torch.float32]
'''
def load_lora_model(base_model_name, output_dir, device_map='cuda', torch_dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map=device_map, torch_dtype=torch_dtype)
    return tokenizer, model


