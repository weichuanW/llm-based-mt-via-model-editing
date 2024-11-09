import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from Support_Utils.BasicReading import BasicReading as BR
from Support_Utils.BasicStorage import BasicStorage as BS
import torch
from transformers import AutoTokenizer
from Support_Utils.model_loading import *
from Support_Utils.FV_Data_Loader import *
from Support_Utils.model_inner_extraction import *
from Support_Utils.model_inner_intervention import *

set_seed(42)
# load the model
error_data_root = '/mnt/sgnfsdata/tolo-02-95/weicwang/shared_llm_mt/Data/Improved_Adaptation/FV_Error'
langs = ['en-de', 'de-en', 'en-zh', 'zh-en']
model_name = '/mnt/sgnfsdata/tolo-02-95/weicwang/model/llama2-7b-hf'
device='cuda:1'
model, tokenizer, model_config = load_model_tokenizer_config(model_name, device=device)

data_ome_shot = '../Data/Source_Data/One_Shot_Cases'
data_zero_shot = '../Data/Source_Data/Zero_Shot_Cases'
training_root = 'Training_Set'
testing_root = 'Test_Set'

# load the data
zero_shot_files = [f'zero_shot_{lang}.json' for lang in langs]
one_shot_files = [f'one_shot_{lang}.json' for lang in langs]

# load the data
