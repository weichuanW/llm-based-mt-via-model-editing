import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import random


'''
D: load the model, tokenizer and model config for inner representation
I: model_name[str], device[str] (default is cuda), cache_dir[str](default is None)
O: model, tokenizer, MODEL_CONFIG
'''
def load_model_tokenizer_config(model_name: str, device='cuda', cache_dir=None, torch_type_=None):
    assert model_name is not None

    print("Loading: ", model_name)

    if 'llama' in model_name.lower():
        if '70b' in model_name.lower():
            # use quantization. requires `bitsandbytes` library
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                device_map=device,
                quantization_config=bnb_config
            )
        else:
            if '7b' in model_name.lower():
                model_dtype = torch.float32
            elif '8b' in model_name.lower():
                model_dtype = torch.float32
            elif '13b' in model_name.lower():
                model_dtype = torch.float32
            else:  # half precision for bigger llama models
                model_dtype = torch.float16
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            if torch_type_:
                model_dtype = torch_type_
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map=device, torch_dtype=model_dtype)

        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config._name_or_path,
                        "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "layer_hook_names": [f'model.layers.{layer}' for layer in
                                             range(model.config.num_hidden_layers)],
                        "KN_hook_names": [f'model.layers.{layer}.mlp.down_proj' for layer in
                                          range(model.config.num_hidden_layers)],
                        'kn_dim': model.config.intermediate_size,
                        }
    else:
        raise NotImplementedError("Still working to get this model available!")

    return model, tokenizer, MODEL_CONFIG


def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)