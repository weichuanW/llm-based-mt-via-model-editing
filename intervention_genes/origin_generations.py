import os
import sys

from past.translation import hooks

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from Support_Utils.BasicStorage import BasicStorage as BS
from Support_Utils.BasicReading import BasicReading as BR
from Support_Utils.model_loading import *
from Support_Utils.model_generation import PromptGenerator as PG
from Support_Utils.model_inner_intervention import *
from Evals.eval_test import *
import argparse


# intervention stage
'''
D: generation for suppress the heads    
I: data_cases: [list[str]], each dict contains the prompt
-- layer_head_token_pairs: [list[tuple]], each tuple contains the (layer, head, token)
-- model: [transformer model], the model to be used, tokenizer: [transformer tokenizer], model_config: [dict]
-- stop_words: [list[str]], stop_num: [int], corresponding to shot e.g. for zero-shot, with ['\n'], 2
-- batch_size: [int], max_length: [int], head_hook_layer: [str], avg_activations: [tensor] (default is None for suppress and enhance)
-- operation: [str], suppress or enhance or others (must with avg vector)
O: filtered_generation: [list[str]], the generation after intervention without prompt
'''
def ori_genes(data_cases, model, tokenizer, model_config, stop_words, stop_num, batch_size, max_length):

    generation = pg_.batch_generation(data_cases, model, tokenizer, stop_words, stop_num, model.device, batch_size, max_length)
    # filter the prompt
    filtered_generation = [generation[i].replace(data_cases[i], '').strip().split('\n')[0].strip() for i in range(len(generation))]
    return filtered_generation



# evaluation stage
'''
D: evaluate the cases with language recognition and repetition recognition
I: data: [list[dict]], each dict contains the source, target and generation
-- tokenizer: [transformer tokenizer], lang: [str], max_token: [int]
O: data: [list[dict]], each dict contains the source, target, generation, lang_detect, repetition, repetition_str, repetition_retrieval_word
-- set_metrics: [dict], the metrics for the set, including lang_acc, repe_ratio, BLEU, COMET22DA, COMET22KIWI
'''
def evaluate_cases(data, tokenizer, lang, max_token):
    # language recognition
    # case updating
    source_ = [item['source'] for item in data]
    target_ = [item['target'] for item in data]
    prediction_ = [item['generation'] for item in data]

    language_results = language_recognise(prediction_, tool='polyglot', exp_lang=lang)
    language_detect_info = language_results['details']
    repetition_all, repetition_strs = repe_recognise(prediction_, tokenizer, lang, max_token)

    # case updating for language
    for i, item in enumerate(language_detect_info):
        data[i]['lang_detect'] = language_detect_info[i]

    # case updating for repetition
    repe_ids = [item[0] for item in repetition_strs]

    for i, item in enumerate(repetition_strs):
        repe_id = item[0]
        data[repe_id]['repetition'] = 1
        data[repe_id]['repetition_str'] = [item[1], item[2]]
        data[repe_id]['repetition_retrieval_word'] = [repetition_all[i][1], repetition_all[i][2]]

    # update the repetition all
    for i, item in enumerate(data):
        if i not in repe_ids:
            data[i]['repetition'] = 0
            data[i]['repetition_str'] = ''
            data[i]['repetition_retrieval_word'] = ''

    # set updating
    if lang!='zh':
        lang = '13a'
    mt_metrics_all = mt_metrics(sources=source_, predictions=prediction_, references=target_, tokenize=lang)
    set_metrics = dict()
    set_metrics['lang_acc'] = sum(language_detect_info)/len(language_detect_info)
    set_metrics['repe_ratio'] = sum([item['repetition'] for item in data])/len(data)
    set_metrics['BLEU'] = mt_metrics_all['SacreBLEU']
    set_metrics['COMET22DA'] = mt_metrics_all['22cometda']
    set_metrics['COMET22KIWI'] = mt_metrics_all['22cometkiwi']
    return data, set_metrics

# real testing
'''
D: official processing for a specific setting, processing the generation and evaluation
I: layer_head_token_pairs: [list[tuple]], each tuple contains the (layer, head, token)
-- subset_root: [str], subset_root_save: [str], test_file: [str]
-- head_type: [str] (e.g. fv-trace-error, fv-task-heads)
-- model: [transformer model], tokenizer: [transformer tokenizer], model_config: [dict]
-- topk: [int], lang_target: [str], batch_size: [int], max_length: [int]
-- operation: [str], suppress or enhance or others (if others, must with avg vector)
O: eval_set_results: [dict], the evaluation results for the set
'''
def processing(subset_root, subset_root_save, file_type, test_file, head_type, model, tokenizer, model_config, topk, lang_target, batch_size, max_length,operation='none'):
    data = BR(os.path.join(subset_root, test_file)).data_reader['json']()
    if len(data) == 0:
        print('The data is empty for file!'.format(test_file))
        return 0
    sorted_data = sorted(data, key=lambda x: x['prompt'], reverse=True)
    sorted_cases = [item['prompt'] for item in sorted_data]
    if os.path.exists(os.path.join(subset_root_save, f'{head_type}_top-{topk}-head_{operation}_gene_{file_type}' + test_file)):
        print('The file has been processed!'.format(test_file))
        sorted_data = BR(os.path.join(subset_root_save, f'{head_type}_top-{topk}-head_{operation}_gene_{file_type}' + test_file)).data_reader['json']()
    else:

        if 'one_shot' in test_file:
            stop_words = ['\n']
            stop_num = 4
            genes = ori_genes(sorted_cases, model, tokenizer, model_config,
                                             stop_words=stop_words, stop_num=stop_num, batch_size=batch_size,
                                             max_length=max_length)
        elif 'zero_shot' in test_file:
            stop_words = ['\n']
            stop_num = 2
            genes = ori_genes(sorted_cases, model, tokenizer, model_config,
                                             stop_words=stop_words, stop_num=stop_num, batch_size=batch_size,
                                             max_length=max_length)
        else:
            raise ValueError('The file name is not correct!')
            # merge generation to original cases
        for i, item_case in enumerate(sorted_data):
            item_case['generation'] = genes[i]
        # storage for generation
        storage_gene_name = os.path.join(subset_root_save, f'{head_type}_top-{topk}-head_{operation}_gene_{file_type}' + test_file)
        BS(sorted_data, storage_gene_name).data_reader['json']()
    # storage for detection
    eval_detail, eval_set_results = evaluate_cases(sorted_data, tokenizer, lang_target, max_length)
    eval_set_results['file'] = test_file
    storage_eval_name = os.path.join(subset_root_save, f'{head_type}_top-{topk}-head_{operation}_detect_{file_type}' + test_file)
    BS(eval_detail, storage_eval_name).data_reader['json']()
    return eval_set_results
# preparation

if __name__ == '__main__':
    # original generations with lang prompt without any intervention

    parser = argparse.ArgumentParser()
    parser.add_argument('--one_shot_root', help='the dataset for one-shot ', type=str, required=True)
    parser.add_argument('--zero_shot_root', help='the dataset for zero-shot', type=str, required=True)
    parser.add_argument('--lang_pairs', help='specific testing language pairs', type=str, required=True)
    parser.add_argument('--subset_root_save', help='save for the generation results with intervention', type=str, required=True)
    parser.add_argument('--subset_eval_save', help='save for the evaluation results', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='llama2-7b')
    # local name path or huggingface model name
    # if using local lora model, please directly use the full name with checkpoint
    parser.add_argument('--device', help='Device to run on',type=str, required=True, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', help='batch size for batch generation', type=int, required=True, default=2)
    parser.add_argument('--max_token', help='max generation tokens for new generated content', type=int, required=True, default=400)
    parser.add_argument('--heads_type', help='definition for the intervention prefix', type=str, required=True, default=None)
    parser.add_argument('--file_type', help='file type for the testing or validation', type=str, required=True, default='Test')
    args = parser.parse_args()
    '''
    # example bash
    --one_shot_root '../Data/Source_Data/One_Shot_Cases/Test_set' \
    --zero_shot_root '../Data/Source_Data/Zero_Shot_Cases/Test_set' \
    --lang_pairs 'en-zh' \
    --subset_root_save '../Detect_Data/genes' \
    --subset_eval_save '../Detect_Data/evals' \
    --model_name 'llama2-7b' \
    --device 'cuda' \
    --batch_size 2 \
    --max_token 400 \
    --heads_type 'llama2-7b-ori-genes' \
    --file_type 'Test'
    '''

    one_shot_root = args.one_shot_root
    one_shot_files = os.listdir(one_shot_root)

    zero_shot_root = args.zero_shot_root
    zero_shot_files = os.listdir(zero_shot_root)
    lang_pairs = [str(item) for item in args.lang_pairs.split(',')]
    one_shot_files = [item for item in one_shot_files if item.split('_')[-1].replace('.json', '') in lang_pairs]
    zero_shot_files = [item for item in zero_shot_files if item.split('_')[-1].replace('.json', '') in lang_pairs]

    subset_root_save = args.subset_root_save
    subset_eval_save = args.subset_eval_save
    # default made for real testing
    if not os.path.exists(subset_eval_save):
        os.makedirs(subset_eval_save)

    if not os.path.exists(subset_root_save):
        os.makedirs(subset_root_save)


    # initial model and tokenizer

    model_name = args.model_name
    device = args.device
    model, tokenizer, model_config = load_model_tokenizer_config(model_name, device=device)

    # generator initialise
    pg_ = PG()
    # generation setting
    batch_size = args.batch_size
    max_length = args.max_token
    operation = 'none'



    head_type = args.heads_type
    top_k = 0
    eval_files = []
    overall_set_eval = list()
    file_type = 'Test'
    if os.path.exists(os.path.join(subset_eval_save, f'{head_type}_top-{top_k}-head_{operation}_eval_one-shot_{file_type}_all.json')):
        print('The evaluation is already processed!')
        overall_set_eval = BR(os.path.join(subset_eval_save, f'{head_type}_top-{top_k}-head_{operation}_eval_one-shot_{file_type}_all.json')).data_reader['json']()
        eval_files = [item['file'] for item in overall_set_eval]

    # one-shot
    for test_file in one_shot_files:
        file_items = test_file.split('_')
        lang = file_items[-1].replace('.json', '')
        lang_target = lang.split('-')[-1]

        if test_file in eval_files:
            print('The file has been processed for eval!'.format(test_file))
            continue

        # aligned with the mean activation files

        set_result = processing(one_shot_root, subset_root_save,file_type, test_file, head_type, model, tokenizer, model_config, top_k, lang_target, batch_size, max_length)
        overall_set_eval.append(set_result)
        eval_files.append(test_file)
        # storage the evaluation results
        storage_eval_name = os.path.join(subset_eval_save, f'{head_type}_top-{top_k}-head_{operation}_eval_one-shot_{file_type}_all.json')
        BS(overall_set_eval, storage_eval_name).data_reader['json']()

    eval_files = []
    overall_set_eval = list()
    file_type = 'Test'
    if os.path.exists(os.path.join(subset_eval_save, f'{head_type}_top-{top_k}-head_{operation}_eval_zero-shot_{file_type}_all.json')):
        print('The evaluation is already processed!')
        overall_set_eval = BR(os.path.join(subset_eval_save, f'{head_type}_top-{top_k}-head_{operation}_eval_zero-shot_{file_type}_all.json')).data_reader['json']()
        eval_files = [item['file'] for item in overall_set_eval]

    # zero-shot
    for test_file in zero_shot_files:
        file_items = test_file.split('_')
        lang = file_items[-1].replace('.json', '')
        lang_target = lang.split('-')[-1]

        if test_file in eval_files:
            print('The file has been processed for eval!'.format(test_file))
            continue



        set_result = processing(zero_shot_root, subset_root_save, file_type, test_file, head_type, model, tokenizer, model_config, top_k, lang_target, batch_size, max_length)
        overall_set_eval.append(set_result)
        eval_files.append(test_file)
        # storage the evaluation results
        storage_eval_name = os.path.join(subset_eval_save, f'{head_type}_top-{top_k}-head_{operation}_eval_zero-shot_{file_type}_all.json')
        BS(overall_set_eval, storage_eval_name).data_reader['json']()