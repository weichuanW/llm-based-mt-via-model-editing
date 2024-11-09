from BasicReading import BasicReading as BR
from BasicStorage import BasicStorage as BS
from model_loading import *
from FV_Data_Loader import *
from model_inner_extraction import *
from model_generation import *
from model_inner_intervention import *
from model_IG_compute import *
from baukit import TraceDict

# storage test
read_file = 'test.json'
a = [1, 2, 3, 4, 5]
BS(a, read_file).data_reader['json']()

# reading test
data = BR(read_file).data_reader['json']()

# model loading

model_name = '/mnt/sgnfsdata/tolo-02-95/weicwang/model/llama2-7b-hf'
device_map = 'cuda:3'
cache_dir = ''
model, tokenizer, model_config = load_model_tokenizer_config(model_name, device_map, cache_dir)

# dataset loading
dataset_name = 'mt_de-en.json'
dataset = load_dataset(dataset_name)


# attn_extraction
prompt = 'I have a dream on the'

token_indices = [-1] # only extract the last one
#attn_extra = get_mean_head_activations_batch(dataset, model, model_config, tokenizer, token_indices, batch_size=1)

# kn extraction
#kn_extra = get_mean_mlp_activations_kn_batch(dataset, model, model_config, tokenizer, token_indices, batch_size=1, kn_hook_name='KN_hook_names')

# layer extraction
#layer_extra = get_mean_layer_activations_batch(dataset, model, model_config, tokenizer, token_indices, batch_size=1, layer_name='layer_hook_names')

# generation
pg = PromptGenerator()
batch_data = [prompt]
stop_words=['\n']
stop_num=2
device='cuda:3'
batch_size=4
max_length=400
b = pg.batch_generation(batch_data, model, tokenizer, stop_words, stop_num, device, batch_size, max_length)

# IG computation
replace_module = 'KN_hook_names'
b = internal_integral(prompt, 22789, model, tokenizer, model_config, layers=32, replace_module=replace_module, batch_size=1,
                  steps=20)
print(b.shape)
# KN intervention

edit_dict = {1:[223, 243]}
mode = 'suppress'
intervention_fn = edit_kn(edit_dict, model, model_config, model.device, mode=mode, idx=-1)
nll_inputs = tokenizer(prompt, return_tensors='pt').to(device)

with TraceDict(model, layers=model_config['KN_hook_names'], edit_output=intervention_fn):
    print('running on this stage')
    model(**nll_inputs)

one_file = ''
data = BR(one_file).data_reader['json']()

data = sorted(data, key = lambda item: len(item['prompt']), reverse=True)
data_input = [data_['prompt'] for data_ in data]


# {'input':  'output':}
# 1, 2, 1,' 1, 1, 1 ,1 ,1 ' 10