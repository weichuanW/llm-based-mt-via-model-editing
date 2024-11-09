from .eval_test import *


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