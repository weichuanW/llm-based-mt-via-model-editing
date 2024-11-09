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
from baukit import TraceDict
from tqdm import tqdm
# running logic
# extraction the error generation cases and construct the error dataset
# error token position identification for error FV computation
# check the influence of error FV by change the repetition span (first repe token) on extraction cases (remove operation)
# generate the error FV and remove it
'''
D: remove the elements of tensor1 from tensor2
'''
def remove_tensor_elements(tensor1, tensor2):
    # Convert tensors to lists
    list1 = tensor1.tolist()
    list2 = tensor2.tolist()

    # Remove elements of list1 from list2
    for item in list1:
        if item in list2:
            list2.remove(item)

    # Convert the result back to a tensor
    result = torch.tensor(list2)

    return result

def error_data_construct(filtered_data, save_root):
    final_storage = list()
    for i, item in enumerate(filtered_data):
        temp_store = dict()
        ori_prompt_ = item['prompt']
        gene_prompt = item['repetition_str'][0]

        overall_genes = ori_prompt_ + item['generation']
        normal_prompt = ori_prompt_ + gene_prompt
        repetition_str = item['repetition_str'][1]

        if (normal_prompt + repetition_str) in overall_genes:
            space = ''
        if (normal_prompt + ' ' + repetition_str) in overall_genes:
            space = ' '
        # input is the normal generation add the first repe
        input = normal_prompt + space + repetition_str

        # output is the rest repe substring
        output = space + repetition_str
        temp_store['input'] = input
        temp_store['output'] = output
        final_storage.append(temp_store)
    return final_storage


'''
D: count the token length for token id extraction
'''
def error_prompt_pairs(filtered_data, tokenizer, max_token_num=1024):
    pairs = list()
    counting_list = list()
    for i, item in enumerate(filtered_data):
        temp_store = dict()
        ori_prompt_ = item['prompt']
        gene_prompt = item['repetition_str'][0]

        overall_genes = ori_prompt_ + item['generation']
        normal_prompt = ori_prompt_ + gene_prompt
        repetition_str = item['repetition_str'][1]

        # determine the concatenation signal
        # we add this signal since our design on repetition string detection is based on separated words rather than tokens
        if (normal_prompt + repetition_str) in overall_genes:
            space = ''
        if (normal_prompt + ' ' + repetition_str) in overall_genes:
            space = ' '

        # compute the initial token list for repe str
        prompt_ids = tokenizer(normal_prompt, return_tensors="pt")['input_ids'][0]
        overall_ids = tokenizer(normal_prompt + space + repetition_str, return_tensors="pt")['input_ids'][0]

        # repe strings
        selection_ids = remove_tensor_elements(prompt_ids, overall_ids)
        temp_store['prompt_len'] = len(overall_ids)
        temp_store['repe_len'] = len(selection_ids)
        pairs.append(temp_store)
    # random selecting the instance
    return pairs

# local construction for temp usgae
flag =0
if flag == 'construct_data':
    # initial the tokenizer
    model_name = '/mnt/sgnfsdata/tolo-02-95/weicwang/model/llama2-7b-hf'
    # load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # extract the error cases
    root_dir = '/mnt/sgnfsdata/tolo-02-95/weicwang/shared_llm_mt/Detect_Results/Original_Gene/Training_set'
    files = os.listdir(root_dir)
    one_shots = [file for file in files if 'one_shot' in file]


    save_root = '/mnt/sgnfsdata/tolo-03-97/weicwang/shared_llm_mt/Data/Improved_Adaptation/FV_Error'
    for file in one_shots:
        data = BR(os.path.join(root_dir, file)).data_reader['json']()
        lang = file.split('_')[-1].split('.')[0]
        # filter the repetition
        filter_data = [item for item in data if item['repetition'] == 1]
        if len(filter_data) == 0:
            print(f'No error data in {file}')
            continue
        # construct the error data
        error_data = error_data_construct(filter_data, '')
        # construct the error position
        pairs = error_prompt_pairs(filter_data, tokenizer, max_token_num=1024)
        BS(error_data, os.path.join(save_root, f'fv_error_data_{lang}.json')).data_reader['json']()
        BS(pairs, os.path.join(save_root, f'fv_error_positions_{lang}.json')).data_reader['json']()

        # 10 repes
        error_10_data = [{'input':data['input']+data['output']*10, 'output':data['output']} for data in error_data]
        pair_10 = [(data['prompt_len'], data['prompt_len'] + data['repe_len']*10) for data in pairs]
        BS(error_10_data, os.path.join(save_root, f'fv_error_data_10_{lang}.json')).data_reader['json']()
        BS(pair_10, os.path.join(save_root, f'fv_error_positions_10_{lang}.json')).data_reader['json']()
        print('111')
        save_root = '/mnt/sgnfsdata/tolo-03-97/weicwang/shared_llm_mt/Data/Improved_Adaptation/FV_Error'
        for file in one_shots:
            data = BR(os.path.join(root_dir, file)).data_reader['json']()
            lang = file.split('_')[-1].split('.')[0]
            # filter the repetition
            filter_data = [item for item in data if item['repetition'] == 1]
            if len(filter_data) == 0:
                print(f'No error data in {file}')
                continue
            # construct the error data
            error_data = error_data_construct(filter_data, '')
            # construct the error position
            pairs = error_prompt_pairs(filter_data, tokenizer, max_token_num=1024)
            BS(error_data, os.path.join(save_root, f'fv_error_data_{lang}.json')).data_reader['json']()
            BS(pairs, os.path.join(save_root, f'fv_error_positions_{lang}.json')).data_reader['json']()

# official error FV extraction
data_root = '/mnt/sgnfsdata/tolo-03-97/weicwang/shared_llm_mt/Data/Improved_Adaptation/FV_Error'
langs = ['en-de', 'de-en', 'en-zh', 'zh-en']
model_name = '/mnt/sgnfsdata/tolo-02-95/weicwang/model/llama2-7b-hf'
device='cuda:1'
model, tokenizer, model_config = load_model_tokenizer_config(model_name, device=device)

def transfer_to_dict(data_):
    new_dict = dict()
    for item in data_:
        for name in item:
            if name not in new_dict:
                new_dict[name] = list()
            new_dict[name].append(item[name])
    return new_dict

'''
D: get the expected token of the first token
I: normal_prompts: the normal prompts [list(str)], repe_prompts: the repetition prompts [list(str)]
O: the first token ids [list(int)]
'''
def get_first_error_token(normal_prompts, repe_prompts):
    first_ids = list()
    for i, item in enumerate(normal_prompts):
        normal_prompt = item
        repe_prompt = repe_prompts[i]
        prompt_ids = tokenizer(normal_prompt, return_tensors="pt")['input_ids'][0]
        overall_ids = tokenizer(normal_prompt + repe_prompt, return_tensors="pt")['input_ids'][0]

        # repe strings
        selection_ids = remove_tensor_elements(prompt_ids, overall_ids)
        first_id = selection_ids[0]
        first_ids.append(first_id)
    return first_ids

'''
D: get the causal indirect effect of error function vector with the token
I: cases: list of error cases contain the normal generation and the first repetition [list(str)]
-- model: the model for the computation [transformers model], model_config: the model configuration [dict]
-- avg_activations: the average head activations [torch.tensor], token_id_of_interest: the token id of interest to compute the effect [list(int)]
'''
def CIE_compute_error_FV_token(cases, model, model_config, avg_activations, token_id_of_interest, operation='subtract', device='cuda'):
    # error extraction
    # CIE computation
    n_trials = len(cases)
    indirect_effect = torch.zeros(n_trials, model_config['n_layers'], model_config['n_heads'])
    for i, case in enumerate(cases):
        inputs = tokenizer(case, return_tensors='pt').to(device)

        indirect_effect_storage = torch.zeros(model_config['n_layers'], model_config['n_heads'], 1)

        # original prob
        clean_output = model(**inputs).logits[:, -1, :].detach().cpu()
        clean_probs = torch.softmax(clean_output[0], dim=-1)
        for layer in tqdm(range(model_config['n_layers']), desc='Layer effect computation'):
            head_hook_layer = [model_config['attn_hook_names'][layer]]
            for head_n in tqdm(range(model_config['n_heads']), desc='Head effect computation'):
                layer_head_token_pairs = [(layer, head_n, -1)]
                intervention_fn = inter_activation_attn_head(layer_head_token_pairs, avg_activations, model, model_config, act_replace_id=-1,
                                            last_token_only=True, operations=operation, device=device)
                with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:
                    output = model(**inputs).logits[:, -1, :].detach().cpu()  # batch_size x n_tokens x vocab_size, only want last token prediction

                    # TRACK probs of tokens of interest
                intervention_probs = torch.softmax(output, dim=-1)  # convert to probability distribution
                indirect_effect_storage[layer, head_n] = (intervention_probs - clean_probs).index_select(1,torch.LongTensor(token_id_of_interest[i]).squeeze()).squeeze()
                print(111)
        indirect_effect[i] = indirect_effect_storage.squeeze()

    return indirect_effect

for lang in langs:
    # using ten repetitions for each case
    if os.path.exists(os.path.join(data_root, f'fv_error_data_10_{lang}.json')):
        data = BR(os.path.join(data_root, f'fv_error_data_10_{lang}.json')).data_reader['json']()
        # transfer to dict
        data = transfer_to_dict(data)
        positions = BR(os.path.join(data_root, f'fv_error_positions_10_{lang}.json')).data_reader['json']()
        # transfer positions to tuple
        positions = [(item[0], item[1]) for item in positions]
        datasets_lang = load_dataset(data, test_size=None)
        # error extraction
        print('error cases for {} are {}'.format(lang, len(data['input'])))
        # np.array to save memory, please transfer it to tensor for intervention
        if os.path.exists(os.path.join(data_root, f'fv_error_meanheads_{lang}.pt')):
            mean_activation = BR(os.path.join(data_root, f'fv_error_meanheads_{lang}.pt')).data_reader['pt']()
        mean_activation = get_mean_head_activations_batch(datasets_lang, model, model_config, tokenizer, positions, batch_size=1)

        # transfer np array to tensor
        mean_activation = torch.tensor(mean_activation)
        # mean storage
        BS(mean_activation, os.path.join(data_root, f'fv_error_meanheads_{lang}.pt')).data_reader['pt']()
        interven_data = BR(os.path.join(data_root, f'fv_error_data_{lang}.json')).data_reader['json']()
        inter_prompts = [item['input'] for item in interven_data]
        inter_outputs = [item['output'] for item in interven_data]
        first_tokens = get_first_error_token(inter_prompts, inter_outputs)
        if os.path.exists(os.path.join(data_root, f'fv_error_cie_{lang}.pt')):
            print(f'Error FV for {lang} has been computed')
            continue
        indirect_effect = CIE_compute_error_FV_token(inter_prompts, model, model_config, mean_activation, first_tokens, operation='subtract', device=device)
        # cie storage
        BS(indirect_effect, os.path.join(data_root, f'fv_error_cie_{lang}.pt')).data_reader['pt']()

