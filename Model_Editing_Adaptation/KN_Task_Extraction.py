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
D: create the folder
I: folder_name [str], path [str]
'''
def create_folder(folder_name, path = None):
    create_folder_ = os.path.join(path, folder_name)
    if not os.path.isdir(create_folder_):
        os.makedirs(create_folder_)
    else:
        pass

'''
D: filter the normal generation based on our language detector and repetition detector
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
        # we filter the normal generation based on our language detector and repetition detector
        if item['lang_detect'] == 1 and item['repetition'] == 0:
            filter_normal_data.append(item)
        else:
            pass

    return filter_normal_data


'''
D: remove the elements of tensor1 from tensor2, used for generation token extraction
I: tensor1 [torch.tensor], tensor2 [torch.tensor]
O: result [torch.tensor]
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
D: we use tokenizer to random choose the position and do not need the target language
I: filtered_data [list(dict)], tokenizer [AutoTokenizer], instance_num [int], random_num [int], max_random [int]
O: construction_data [list(dict)]
'''
def task_prompt_construct(filtered_data, tokenizer, instance_num=2000, random_num=6, max_random=20):
    construction_data = list()
    # overall collection
    for item in filtered_data:
        gene_ = item['generation']
        prompt_ = item['prompt']
        overall_gene = prompt_ + gene_
        prompt_ids = tokenizer(prompt_, return_tensors="pt")['input_ids'][0]
        overall_ids = tokenizer(overall_gene, return_tensors="pt")['input_ids'][0]

        selection_ids = remove_tensor_elements(prompt_ids, overall_ids)
        selection_ids = selection_ids.tolist()
        # random selection
        for i in range(min(random_num, len(selection_ids))):
            random_pos = random.choice(range(min(max_random, len(selection_ids))))
            selection_random = selection_ids[:random_pos]
            selection_random = torch.tensor(selection_random)
            # concate them together
            random_prompt = torch.cat((prompt_ids, selection_random)).long()
            length = len(random_prompt)
            attn_mask_random = torch.ones(length)
            attn_mask_random = attn_mask_random.long()
            # extend them with additional batch dimension on the first dimension
            random_prompt = random_prompt.unsqueeze(0)
            attn_mask_random = attn_mask_random.unsqueeze(0)
            temp_record = {'prompt': random_prompt, 'attn_mask': attn_mask_random,
                           'exp_token': selection_ids[random_pos]}
            construction_data.append(temp_record)
    # random selecting the instance
    return random.sample(construction_data, instance_num)


'''
D: transfer the data from tensor format to numpy format to store
I: data_ [list(dict)]
O: new_data [list(dict)]
'''
def convert_storage(data_):
    new_data = list()
    for item in data_:
        temp = {'prompt': item['prompt'].tolist(), 'attn_mask': item['attn_mask'].tolist(),
                'exp_token': item['exp_token']}
        new_data.append(temp)
    return new_data


'''
D: sort the data based on the length of the prompt
I: data_ [list(dict)]
O: new_data [list(dict)]
'''
def sort_based_on_length(data_):
    return sorted(data_, key=lambda x: len(x['prompt'][0]), reverse=True)


'''
D: convert the existing data to tensor format, read from the existing file
I: data_ [list(dict)]
'''
def convert_exist_data(data_):
    new_data = list()
    for item in data_:
        temp = {'prompt': torch.tensor(item['prompt']).long(), 'attn_mask': torch.tensor(item['attn_mask']).long(),
                'exp_token': item['exp_token']}
        new_data.append(temp)
    return new_data


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
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for detection')
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

    # load setting for model and tokenizer
    model, tokenizer, model_config = load_model_tokenizer_config(args.model_name, device=args.device,                                                                 cache_dir=args.cache_dir)
    layers = args.total_layers
    detect_module = args.detect_module

    # basic setting for prompt preparation
    instance_num_ = args.instance_num
    random_num_ = args.random_num
    max_random_ = args.max_random

    file_root = args.file_root
    save_root = args.save_root
    lang_pair = args.lang

    # handling file
    file = args.file
    save_folder = f'{args.save_root}/{args.save_name}'
    create_folder(args.save_name, args.save_root)
    overall_kn_data = dict()
    # handling the single data

    pre_lang_set = ['en-de', 'de-en', 'en-zh', 'zh-en']

    if lang_pair not in pre_lang_set:
        raise ValueError('The language pair is not in the pre-defined set')
    if lang_pair not in overall_kn_data:
        overall_kn_data[lang_pair] = ''
    else:
        raise ValueError('The language pair has been handled and some bug exists')
    # filter the data
    # todo: save the filter data
    # check whether the filter data exists
    if f'filter_{instance_num_}_{file}' in os.listdir(save_folder):
        # load the filter data
        print('load the filter data from the existing file')
        filtered_data = BR(f'{save_folder}/filter_{instance_num_}_{file}').data_reader['json']()
    else:
        print('create the filter data')
        filer_name = f'filter_{instance_num_}_' + file
        filtered_data = filter_data_normal(file_root, file)
        # save the filtered data
        # the original data
        BS(filtered_data, f'{save_folder}/normal_{filer_name}').data_reader['json']()

    # prepare the detect prompt
    # for a specific task
    if f'task_prompt_{instance_num_}_{file}' in os.listdir(save_folder):
        print('load the task prompt from the existing file')
        task_prompt = convert_exist_data(
            BR(f'{save_folder}/task_prompt_{instance_num_}_{file}').data_reader['json']())
        task_prompt = sort_based_on_length(task_prompt)
    else:
        print('create the task prompt')
        task_prompt = task_prompt_construct(filtered_data, tokenizer, instance_num=instance_num_,
                                            random_num=random_num_, max_random=max_random_)
        BS(convert_storage(task_prompt),
              f'{save_folder}/task_prompt_{instance_num_}_{file}').data_reader['json']()

    # handling the KN data and record all the data
    for file_ in [file]:
        temp_out = ''
        # we store the all KN activation attribution summations for the model
        json_file = file.replace('.json', '.npy')
        # check for temp storage
        if os.path.exists(f'{save_folder}/kn_attr_{instance_num_}_{instance_num_}_{json_file}'):
            print('The file has been handled {}'.format(json_file))
            break
        # check for a specific setting storage
        if os.path.exists(f'{save_folder}/kn_attr_{instance_num_}_{json_file}'):
            print('The file has been handled {}'.format(json_file))
            break
        for j, item in tqdm(enumerate(task_prompt), desc='KN detection on task KNs'):
            if j > instance_num_:
                break

            # each 500 cases, we do a temp staorage
            check_num = (j // 500) * 500

            # construction for model input
            construct_input = {'input_ids': item['prompt'], 'attention_mask': item['attn_mask']}
            exp_token_id = item['exp_token']
            out = internal_integral(construct_input, exp_token_id, model, tokenizer, model_config, layers=layers,
                                    replace_module=args.detect_module, batch_size=args.batch_size, steps=args.steps)
            # transfer out to numpy
            out = out.detach().cpu().numpy()
            if len(temp_out) == 0:
                temp_out = out
            else:
                temp_out += out

            # convert the tensor to numpy and do storage
            if j % 500 == 0:
                BS(temp_out,
                      f'{save_folder}/kn_attr_temp/kn_attr_{j}_{instance_num_}_{json_file}').data_reader[
                    'npy']()
        # temp_out = temp_out.detach().cpu().numpy()
        # json_file = file.replace('.json', '.npy')
        BS(temp_out, f'{save_folder}/kn_attr_{instance_num_}_{json_file}').data_reader['npy']()


