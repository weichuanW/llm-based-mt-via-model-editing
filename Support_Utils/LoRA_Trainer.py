import os
import sys
'''
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import wandb
from Events.WandbLogging import WandbLogging as WB
'''


from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

class PEFTTrainer(object):
    def __init__(self):
       pass


    def initial_config(self, mode):
        if mode == '4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif mode =='8':
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=200.0,
            )
        else:
            raise ValueError('mode should be 4 or 8')

        return bnb_config


    def base_model_loading(self, base_model_name, bnb_config, cache_path, device_map='cuda', type='peft'):
        if type == 'peft':
            device_map = {'': 0}
            print('quantization model is running')
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                use_auth_token=True,
                cache_dir=cache_path
            )
            base_model.config.use_cache = False
            base_model.config.pretraining_tp = 1
        elif type == 'llm':
            device_map = {'': 0}
            print('original model is running')
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map=device_map,
                trust_remote_code=True,
                use_auth_token=True,
                cache_dir=cache_path
            )
            base_model.config.use_cache = False
            base_model.config.pretraining_tp = 1
        else:
            raise ValueError('type should be peft or llm')
        return base_model

    def load_tokenizer(self, base_model_name, cache_path):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, cache_dir=cache_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    '''
    D: initial the lora config for peft
    I: None
    O: the lora config [LoraConfig]
    *N: "llama": ["q_proj", "v_proj"],
    '''
    def initial_lora(self):
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return peft_config

    '''
    D: load the dataset for training
    I: the dataset name [str]
    O: the dataset [dataset]
    *N: specifically, the dict format is {'data':[{'text': value}, {'text': value}]}
    '''
    def load_dataset(self, name_json):
        return load_dataset("json", data_files=name_json, field="data", split="train")

    '''
    D: the general trainer for PEFT tuning
    I: the name of the base model [str], the mode of the peft tuning [int], the cache path [str] (must provide), 
    -- the output dir for peft model storage[str], the save name for the specific training[str], the dataset name for training[str],
    -- the device type for training [str(default is cuda)], the batch size [int(default is 4)], 
    -- the max training length for input sequence [int](default is 1024), the gradient accumulation steps [int(default is 4)]
    -- the learning rate [float(default is 2e-4)], the logging steps [int(default is 10, we recommend the 1/50 of the overall training)], 
    -- the max steps [int(default is 500)], the neftune noise [int(default is 5)]
    O: the storage of the peft model [model]
    *N: we do not provide evaluation in this process since we hope the model can learn the MT task rather than achieving high performance
    -- the detailed frame may update later with a eval dataset support
    '''
    def _trainer(self, base_model_name, mode, cache_path, output_dir, save_name, dataset_name, device_map='cuda', batch_size=4, max_seq_length=1024, gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=10, save_checkpoints=1, epoches=1, savesteps=0.2, neftune_noise=5, type_='peft'):
        # initialisation basic configs
        bnb_config = self.initial_config(mode)
        base_model = self.base_model_loading(base_model_name, bnb_config, cache_path, device_map, type=type_)
        dataset = self.load_dataset(dataset_name)
        peft_config = self.initial_lora()
        tokenizer = self.load_tokenizer(base_model_name, cache_path)

        # initial the remote logging for wandb

        output_dir = os.path.join(output_dir, save_name)
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_total_limit=save_checkpoints,  # 80 in 80G save_checkpoints, epoches, savesteps
            num_train_epochs=epoches,
            save_steps=savesteps
        )

        trainer = SFTTrainer(
            model=base_model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            neftune_noise_alpha=neftune_noise,
        )
        # neftune_noise_alpha=neftune_noise,

        trainer.train()


