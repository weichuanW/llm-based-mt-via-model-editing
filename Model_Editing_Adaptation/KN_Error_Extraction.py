import os
import sys

# add the current path into system path
project_current = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_current not in sys.path:
    sys.path.append(project_current)

from Support_Utils.model_IG_compute import *
from Support_Utils.model_loading import *
from Support_Utils.BasicStorage import BasicStorage as BS
from Support_Utils.BasicReading import BasicReading as BR

import argparse
import torch
from tqdm import tqdm

'''
 D: filter the repetition generation based on our repetition detector
 I: root_path for all detection handling files[str], file_ [str]
 D: the filtered normal generation data under the one-shot LLM-MT [list] 
 *N: we only do the KN filtering on the training set and one-shot LLM-MT
'''
def filter_data_normal(root_path, file_):
    # only for training set
    assert 'training' in file_
    # only for one-shot LLM-MT
    assert 'one_shot' in file_
    # load the data
    data = BR(f'{root_path}/{file_}').data_reader['json']()
    filter_normal_data = list()
    for item in data:
        if item['repetition'] == 1:
            filter_normal_data.append(item)
        else:
            pass

    return filter_normal_data


'''
D: remove the elements of tensor1 from tensor2
I: tensor1 [tensor], tensor2 [tensor]
O: result [tensor]
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


'''
D: we use tokenizer to build the error token pairs
I: filtered_data [list(dict)], tokenizer [AutoTokenizer], max_token_num [int]
O: filtered pairs [list[dict[tuple]]] (baseline for baseline tokens, repe for repetition tokens)(tensor(size of [k,]), tensor) (prompt, follow repe2)
-- counting_list for extracting error token numbers[list[tuple]] (based on a specific set) (repetition number, repetition string length)
-- for each element, the first element is the prompt, the second element is the repetition tokens
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
        normal_prompt = normal_prompt + space + repetition_str
        # for comparison, we use cat(prompt_ids, selection_ids[i]), cat(cat(prompt_ids, selection_ids), selection_ids[i]))
        # the first prompt without any repeated string
        temp_store['baseline'] = (prompt_ids, selection_ids)
        temp_store['repe_paras'] = list()
        # seaching for overall pairs
        while (normal_prompt + space + repetition_str) in overall_genes:
            # this loop first pair include the normal prompt and the first repeated string
            normal_prompt_id = tokenizer(normal_prompt, return_tensors="pt")['input_ids'][0]
            prompt_pair = (normal_prompt_id, selection_ids)
            temp_store['repe_paras'].append(prompt_pair)
            # update the normal prompt
            normal_prompt = normal_prompt + space + repetition_str
        # filter the repetition pairs with max token setting
        temp_store['repe_paras'] = [item for item in temp_store['repe_paras'] if
                                    len(item[0]) + len(item[1]) < max_token_num]
        # filter the repetition pairs with max repetition setting
        counting_list.append((len(temp_store['repe_paras']), len(selection_ids)))
        pairs.append(temp_store)
    # random selecting the instance
    return pairs, counting_list


'''
D: sort the pairs and counting list based on the counting list
I: counting_list [list[tuple]], pairs [list[dict[tuple]]]
O: sorted_pairs [list[dict[tuple]]], sorted_counting_list [list[tuple]]
'''
def sort_pairs_lists(counting_list, pairs):
    # Get the indices of sorted elements
    indices = sorted(range(len(counting_list)), key=lambda i: counting_list[i][0])

    # Use the indices to sort the pairs
    sorted_pairs = [pairs[i] for i in indices]

    # Use the indices to sort the counting_list
    sorted_counting_list = [counting_list[i] for i in indices]

    return sorted_pairs, sorted_counting_list


'''
D: parallel counting based on the filtered data
I: pairs [list[dict[tuple]]], counting_list [list[tuple]], tokenizer [AutoTokenizer], model [AutoModelForCausalLM], model_config [dict], replace_module [str]
-- batch_size [int], steps [int], max_detect [int], lang_pair [str], filter_data_root [str]
O: overall_storage [list[dict[list]]]
*N: we do a case storage for each case, which includes the baseline and repetition pairs
'''
def parallel_comparison(pairs, counting_list, tokenizer, model, model_config, replace_module, batch_size=5, steps=20, max_detect=10000, save_root=None, save_name=None):
    # based on the counting do the first filtering
    # overall numbers for calculation
    testing_pairs = None
    #tetsing_counting = None

    if len(pairs) == 0:
        storage = list()
        return BS(storage, f'{save_root}/{save_name}_{0}.json').data_reader['json']()
    sorted_pairs, sorted_counting_list = sort_pairs_lists(counting_list, pairs)
    counting = sum([item[0] * item[1] for item in sorted_counting_list])
    if counting <= max_detect:
        testing_pairs = sorted_pairs
        #tetsing_counting = sorted_counting_list
    else:
        sum_all_case_one = sum([item[1] for item in sorted_counting_list])
        length_list = [item[0] for item in sorted_counting_list]
        for item in length_list:
            if item * sum_all_case_one > max_detect:
                testing_pairs = [{'baseline': sorted_case['baseline'], 'repe_paras': sorted_case['repe_paras'][:item]} for sorted_case in sorted_pairs]
                #tetsing_counting = [(item, repe[1]) for repe in sorted_counting_list]
                print('The number of tokens is {}'.format(item * sum_all_case_one))
                break
            else:
                if item * sum_all_case_one < max_detect:
                    continue
    # parallel counting based on the filtered data
    para_pointer = 0 # pointer for the parallel counting initial position
    all_storage = list()
    for para_pointer, item in enumerate(tqdm(testing_pairs, desc='KN detection on error KNs')):
        overall_storage = list()
        # for each element, we have (prompt ids, repe ids) with tensor format of size (k,)
        if os.path.exists(f'{save_root}/{save_name}_{para_pointer}.json'):
            print('The file has been handled {}'.format(para_pointer))
            continue

        baseline = item['baseline']
        repe_paras = item['repe_paras']

        # baseline counting for once and first
        #baseline_construct = {'input_ids': baseline[0].long().unsqueeze(0), 'attention_mask': torch.ones(len(baseline[0])).long().unsqueeze(0)}
        case_store = {'baseline': list(), 'repe_paras': list()}
        for i in tqdm(range(len(baseline[1])), desc='baseline counting for item {}'.format(para_pointer)):
            if i == 0:
                base_prompt_construct = {'input_ids': baseline[0].long().unsqueeze(0), 'attention_mask': torch.ones(len(baseline[0])).long().unsqueeze(0)}
                exp_token_id = baseline[1][i]
                base_para_result = internal_integral(base_prompt_construct, exp_token_id, model, tokenizer, model_config, layers=layers,
                                  replace_module=replace_module, batch_size=batch_size, steps=steps)
            else:
                base_prompt_construct = {'input_ids': torch.cat((baseline[0], baseline[1][:i])).long().unsqueeze(0), 'attention_mask': torch.ones(len(baseline[0]) + i).long().unsqueeze(0)}
                exp_token_id = baseline[1][i]
                base_para_result = internal_integral(base_prompt_construct, exp_token_id, model, tokenizer, model_config, layers=layers,
                                  replace_module=replace_module, batch_size=batch_size, steps=steps)
            values, tuple_indices = top_k_elements(base_para_result, 500)
            tuple_indices = [[int(item[0]), int(item[1])] for item in tuple_indices]
            case_store['baseline'].append(tuple_indices)
            # test
            #bl.BS(case_store, f'{filter_data_root}/test_only.json').data_reader['json']()
            #raise ValueError('test')
        # repetition counting for each case
        for j in tqdm(range(len(repe_paras)), desc='repe counting for item {}'.format(para_pointer)):
            case_store['repe_paras'].append(list())
            for i in tqdm(range(len(repe_paras[j][1])), desc='repe counting for inner item {}'.format(j)):
                if i == 0:
                    base_prompt_construct = {'input_ids': repe_paras[j][0].long().unsqueeze(0), 'attention_mask': torch.ones(len(repe_paras[j][0])).long().unsqueeze(0)}
                    exp_token_id = repe_paras[j][1][i]
                    base_para_result = internal_integral(base_prompt_construct, exp_token_id, model, tokenizer, model_config, layers=layers,
                                      replace_module=replace_module, batch_size=batch_size, steps=steps)
                else:
                    base_prompt_construct = {'input_ids': torch.cat((repe_paras[j][0], repe_paras[j][1][:i])).long().unsqueeze(0), 'attention_mask': torch.ones(len(repe_paras[j][0]) + i).long().unsqueeze(0)}
                    exp_token_id = repe_paras[j][1][i]
                    base_para_result = internal_integral(base_prompt_construct, exp_token_id, model, tokenizer, model_config, layers=layers,
                                      replace_module=replace_module, batch_size=batch_size, steps=steps)
                values, tuple_indices = top_k_elements(base_para_result, 500)
                tuple_indices = [[int(item[0]), int(item[1])] for item in tuple_indices]
                case_store['repe_paras'][j].append(tuple_indices)
        overall_storage.append(case_store)
        BS(overall_storage, f'{save_root}/{save_name}_{para_pointer}.json').data_reader['json']()
        all_storage.append(overall_storage)
    return all_storage



def parse_arguments():
    parser = argparse.ArgumentParser(description='Your script description here.')

    # model setting
    parser.add_argument('--model_name', type=str, default=None, help='The location of the model or model name')
    parser.add_argument('--device', type=str, default='cuda', help='Usage device')
    parser.add_argument('--cache_dir', type=str, default=None, help='The cache_dir for a local model (Optional)')
    parser.add_argument('--detect_module', type=str, default='KN_hook_names',
                        help='detection module name on the model loading config')
    parser.add_argument('--total_layers', type=int, default=32, help='total layer for the model (default: 32)')

    # detection setting
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for detection')
    parser.add_argument('--steps', type=int, default=20, help='steps for the integral')

    # Prompt setting for selection
    parser.add_argument('--instance_num', type=int, default=2000, help='total instance number')
    parser.add_argument('--random_num', type=int, default=6, help='random selection count for each case')
    parser.add_argument('--max_random', type=int, default=20,
                        help='Setting for generation token range, which means we use the first 20 tokens maximum')

    # file setting
    parser.add_argument('--file_root', type=str, default=None, help='The generation file root path')
    parser.add_argument('--file', type=str, default=None,
                        help='The generation file for task KN extraction, must normal generation')
    parser.add_argument('--save_root', type=str, default=None, help='The save root for the detection file')
    parser.add_argument('--save_name', type=str, default=None, help='The save name for the detection file')

    # language setting
    parser.add_argument('--lang', type=str, default=None, help='The language pair for the generation file (e.g. en-de)')

    args = parser.parse_args()
    return args


# basic setting for model and IG
if __name__ == '__main__':
    args = parse_arguments()
    # basic setting for model and IG
    model_name = args.model_name
    device = args.device
    replace_module = args.detect_module
    model, tokenizer, model_config = load_model_tokenizer_config(model_name, device=device, cache_dir=args.cache_dir)
    layers = args.total_layers

    # basic setting for prompt preparation
    instance_num_ = args.instance_num

    batch_size = args.batch_size
    steps = args.steps

    # basic setting for data preparation
    root_path = args.file_root
    file_ = args.file
    save_root = args.save_root
    save_name = args.save_name
    lang_pair = args.lang

    overall_kn_data = dict()
    language_set = ['en-de', 'de-en', 'en-zh', 'zh-en']
    # each file has one unique KN set
    if lang_pair in language_set:
        if lang_pair not in overall_kn_data:
            overall_kn_data[lang_pair] = ''
        else:
            raise ValueError('The language pair has been handled and some bug exists')
        #  testing stage
        filtered_data = filter_data_normal(root_path, file_)
        error_pairs, counting = error_prompt_pairs(filtered_data, tokenizer)
        #BS(error_pairs, f'{save_root}/{save_name}_error_pairs.json').data_reader['json']()
        #BS(counting, f'{save_root}/{save_name}_counting.json').data_reader['json']()
        storage = parallel_comparison(error_pairs, counting, tokenizer, model, model_config, replace_module,
                                      batch_size=batch_size, steps=steps, max_detect=instance_num_, save_root=save_root, save_name=save_name)
        BS(storage, f'{save_root}/{save_name}.json').data_reader['json']()