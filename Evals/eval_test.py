# todo: test the module data
# after finishing testing, writing the a brief readme file
from comet import download_model, load_from_checkpoint
from .general_MT_metrics import MTMetrics as MTM
from .general_langauge_detector import LanguageDetectors as LDs
from .general_repe_detector import RepeRetrieval as RR
from transformers import AutoTokenizer
# detector initial
mtms = MTM()
lang_detector = LDs()
repe_detector = RR()



# BLEU, ChrF,...
# specifically, Chinese is zh, others is 13a
# COMET series
def mt_metrics(sources=None, predictions=None, references=None, tokenize='13a', type=['SacreBLEU', '22cometda','22cometkiwi'], round_=6, batch_size=16, device='cuda'):
    metrics_results = dict()
    if 'SacreBLEU' in type:
        results = mtms.sacrebleu(sources, predictions, references, tokenize)[1]
        metrics_results['SacreBLEU'] = round(results, round_)
    if '22cometda' in type:
        # initial testing model
        model_path = download_model("Unbabel/wmt22-comet-da")  # Unbabel/wmt22-comet-da
        model_comet22qa = load_from_checkpoint(model_path).to(device)
        results = mtms.comet(sources, predictions, references, '22cometda', round_, batch_size, model_comet22qa)[1]
        metrics_results['22cometda'] = round(results, round_)
    if '22cometkiwi' in type:
        local_path_cometkiwi = '/mnt/sgnfsdata/tolo-03-97/weicwang/model/comet-kiwi/checkpoints/model.ckpt'
        model_comet22kiwi = load_from_checkpoint(local_path_cometkiwi).to(device)
        results = mtms.comet(sources, predictions, references, '22cometkiwi', round_, batch_size, model_comet22kiwi)[1]
        metrics_results['22cometkiwi'] = round(results, round_)
    return metrics_results



# langauge detection
def language_recognise(examples, tool=None, exp_lang=None):
    lang_identification = dict()
    lang_identification['acc'] = 0
    lang_identification['details'] = list()
    for example in examples:
        lang_detect = lang_detector.detectors_[tool](example)[0]
        if lang_detect == exp_lang:
            lang_identification['details'].append(1)
        else:
            lang_identification['details'].append(0)
    lang_identification['acc'] = sum(lang_identification['details']) if sum(lang_identification['details'])==0 else sum(lang_identification['details'])/len(lang_identification['details'])
    return lang_identification

'''
D: repetition detector for the model
'''
def repe_recognise(cases, tokenizer,lang, token_setting):
    # repetition detection
    repetition_ids, repetition_retriever, repetition_str, excep = repe_detector.repetition_judgement_retrieval(cases, tokenizer, lang=lang, token_setting=token_setting)
    return repetition_retriever, repetition_str

# test for language detector
'''source = ['I am staying here for some people']
predictions=['我们在这里等人']
references=[['我们在这里等待某些人']]
print(language_recognise(source, 'langdetect', 'en'))
print(mt_metrics(source, predictions, references, 'zh', ['22cometda'], 6, 16, 'cuda'))
# repetition detection
cases = ['I have a dream dream dream dream dream dream dream dream dream']
lang='en'
max_length=10
base_model_name = "/mnt/sgnfsdata/tolo-02-95/weicwang/model/llama2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
print(repe_detector.repetition_judgement_retrieval(cases, tokenizer,lang=lang,token_setting=max_length))'''