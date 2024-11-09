import evaluate
from comet import download_model, load_from_checkpoint
import torch

class MTMetrics(object):
    '''
    D: this class is a basic support for various MT metrics which are generally used in MT evaluation,
    --but not include some basic statistic metrics
    I: metric name [str] (detailed name can be seen in self.metrics_)
    O: related metrics with dictionary [dict] and score [float] (the former is a detailed version)
    E: I: 'bleu', O: {'score': ..}, 22.33
    *N: usage: one type is directly use the score name with object().metrics[scorename](prediction, references)
    --another way is to use the direct function name as the small letter (unified expression)
    '''
    def __init__(self):
        self.metrics = {'SacreBLEU': self.sacrebleu,
                        'BLEU': self.bleu,
                        'NIST': self.nist_mt,
                        'GoogleBLEU': self.google_bleu,
                        'ChrF': self.chrf,
                        'ChrF++': self.chrfplus,
                        'METEOR': self.meteor,
                        'COMET': self.comet,
                        'TER': self.ter
                        }



    '''
    D: SacreBLEU
    I: predictions with list [list[str, str]], references with list of list [list[list[str, str]]]
    O: detailed score dictionary [dict] and score [float]
    E: I: ['I win'], [['I wins', 'You win']], O: {...}, 0.33
    '''
    def sacrebleu(self, sources=None, predictions=None, references=None, tokenize='13a'):
        if not predictions or not references:
            raise ValueError('Please input your predictions/translation results and corresponding reference')

        # change this to the over-all detection
        '''if self.detect_language(references[0][0])=='zh':
            tokenize = 'zh'''

        sacrebleu = evaluate.load("sacrebleu")
        results = sacrebleu.compute(predictions=predictions,
                                    references=references,
                                    tokenize=tokenize)
        score = results["score"]
        return results, score

    '''
    D: BLEU
    I: predictions with list [list[str, str]], references with list of list [list[list[str, str]]]
    O: detailed score dictionary [dict] and score [float]
    E: I: ['I win'], [['I wins', 'You win']], O: {...}, 0.33
    '''
    def bleu(self, sources=None, predictions=None, references=None, tokenize=None):
        if not predictions or not references:
            raise ValueError('Please input your predictions/translation results and corresponding reference')
        bleu = evaluate.load("bleu")
        results = bleu.compute(predictions=predictions,
                                    references=references)
        score = results["bleu"]
        return results, score

    '''
    D: Google BLEU
    I: predictions with list [list[str, str]], references with list of list [list[list[str, str]]]
    O: detailed score dictionary [dict] and score [float]
    E: I: ['I win'], [['I wins', 'You win']], O: {...}, 0.33
    '''
    def google_bleu(self, sources=None, predictions=None, references=None, tokenize=None):
        if not predictions or not references:
            raise ValueError('Please input your predictions/translation results and corresponding reference')
        googlebleu = evaluate.load("google_bleu")
        results = googlebleu.compute(predictions=predictions,
                                    references=references)
        score = results["google_bleu"]
        return results, score
    '''
    D: NIST
    I: predictions with list [list[str, str]], references with list of list [list[list[str, str]]]
    O: detailed score dictionary [dict] and score [float]
    E: I: ['I win'], [['I wins', 'You win']], O: {...}, 0.33
    '''
    def nist_mt(self, sources=None, predictions=None, references=None, tokenize=None):
        if not predictions or not references:
            raise ValueError('Please input your predictions/translation results and corresponding reference')
        nistmt = evaluate.load("nist_mt")
        results = nistmt.compute(predictions=predictions,
                                     references=references)
        score = results["nist_mt"]
        return results, score

    '''
    D: ChrF
    I: predictions with list [list[str, str]], references with list of list [list[list[str, str]]]
    O: detailed score dictionary [dict] and score [float]
    E: I: ['I win'], [['I wins', 'You win']], O: {...}, 0.33
    '''
    def chrf(self, sources=None, predictions=None, references=None, tokenize=None):
        if not predictions or not references:
            raise ValueError('Please input your predictions/translation results and corresponding reference')
        chrf = evaluate.load("chrf")
        results = chrf.compute(predictions=predictions,
                                 references=references)
        score = results["score"]
        return results, score
    '''
    D: ChrF++
    I: predictions with list [list[str, str]], references with list of list [list[list[str, str]]]
    O: detailed score dictionary [dict] and score [float]
    E: I: ['I win'], [['I wins', 'You win']], O: {...}, 0.33
    '''
    def chrfplus(self, sources=None, predictions=None, references=None, tokenize=None):
        if not predictions or not references:
            raise ValueError('Please input your predictions/translation results and corresponding reference')
        chrfplus = evaluate.load("chrf")
        results = chrfplus.compute(predictions=predictions,
                               references=references,
                               word_order=2)
        score = results["score"]
        return results, score
    '''
    D: METEOR
    I: predictions with list [list[str, str]], references with list of list [list[list[str, str]]]
    O: detailed score dictionary [dict] and score [float]
    E: I: ['I win'], [['I wins', 'You win']], O: {...}, 0.33
    '''
    def meteor(self, sources=None, predictions=None, references=None, tokenize=None):
        if not predictions or not references:
            raise ValueError('Please input your predictions/translation results and corresponding reference')
        meteor = evaluate.load("meteor")
        results = meteor.compute(predictions=predictions,
                               references=references)
        score = results["meteor"]
        return results, score

    '''
    D: COMET evaluation for machine translation
    I: source with list [list[str]], predictions with list [list[str]], references with list of list [list[list[str, str]]]
    -- local file for COMET series [str] (suffix is .ckpt), type of COMET [str] (22cometda, 22cometkiwi), etc
    -- float round [int] (default is 6) batch size [int] (default is 16)
    O: case scores [list(float)] and average score [float]
    E: I: ['I win'], ['I wins', 'You win'], O: [0.112323], 0.112323
    *N:This metric need additional data format for original evaluation, details can be found at
    --https://github.com/Unbabel/COMET
    --The format is list[dict{src,mt,ref}] for cometda, which has been done within this function
    -- we did not include specific tokenizer language hinter since the model automatically handles this
    -- the SOTA model is needed to manually download from huggingface, please at least download the model with .ckpt suffix to use
    --e.g. '/PATH/comet-kiwi/checkpoints/model.ckpt' for cometkiwi, PATH is your own prefix of local path
    -- you can modify this code directly with similar logic while adding different model type
    -- one reference is https://huggingface.co/Unbabel/wmt22-cometkiwi-da (download in this page's file block)
    ''' # local_file=None,
    def comet(self, sources=None, predictions=None, references=None, type='22cometda', round_=6, batch_size=16, model=None):
        if type == '22cometda':
            if not sources or not predictions or not references:
                raise ValueError('Please input your predictions/translation results and corresponding reference')
            #model_path = download_model("Unbabel/wmt22-comet-da")#Unbabel/wmt22-comet-da
            #model = load_from_checkpoint(model_path)
            gpus = None
            if gpus is None:
                gpus = 1 if torch.cuda.is_available() else 0
            data = {"src": sources, "mt": predictions, "ref": references}
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            mean_score = model.predict(data, batch_size=batch_size, gpus=gpus)
            case_scores = mean_score['scores']
            case_scores = [round(score_, round_) for score_ in case_scores]
            avg_score = mean_score['system_score']
            avg_score = round(avg_score, round_)
            # gpu free
            '''del model
            torch.cuda.empty_cache()'''
        elif type == '22cometkiwi':
            if not sources or not predictions or not references:
                raise ValueError('Please input your predictions/translation results and corresponding reference')
            '''if not local_file:
                raise ValueError('Please input your local file path for COMET-Kiwi')'''
            gpus = None
            if gpus is None:
                gpus = 1 if torch.cuda.is_available() else 0
            #model = load_from_checkpoint(local_file)
            data = {"src": sources, "mt": predictions}
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            mean_score = model.predict(data, batch_size=batch_size, gpus=gpus)
            case_scores = mean_score['scores']
            case_scores = [round(score_, round_) for score_ in case_scores]
            avg_score = mean_score['system_score']
            avg_score = round(avg_score, round_)
            # gpu free
            '''del model
            torch.cuda.empty_cache()'''
        else:
            raise ValueError('Please input the correct type of COMET model :{}'.format(type))
        return case_scores, avg_score

    '''
    D: TER
    I: predictions with list [list[str, str]], references with list of list [list[list[str, str]]]
    O: detailed score dictionary [dict] and score [float]
    E: I: ['I win'], [['I wins', 'You win']], O: {...}, 0.33
    '''
    def ter(self, sources=None, predictions=None, references=None, tokenize=None):
        if not predictions or not references:
            raise ValueError('Please input your predictions/translation results and corresponding reference')
        meteor = evaluate.load("ter")
        if tokenize == 'zh':
            results = meteor.compute(predictions=predictions,
                                     references=references,
                                     case_sensitive=True,
                                     support_zh_ja_chars=True)
        else:
            results = meteor.compute(predictions=predictions,
                                     references=references,
                                     case_sensitive=True)
        score = results['score']
        return results, score

    '''
    D: output case score to support case evaluation
    I: evaluation method [str], float round [int], source with list [list[str]], predictions with list [list[str]], 
    -- references with list of list [list[list[str, str]]]
    O: case score [list(float)]
    E: I: ['I win'], ['I wins', 'You win'], O: {...}, 0.33
    '''
    def case_evaluate(self, method, float_, sources=None, predictions=None, references=None):
        if method == 'COMET':
            case_score = self.comet(sources, predictions, references)[-1]
            case_score = [round(score_, float_) for score_ in case_score]
        else:
            case_score = list()
            for i, source in enumerate(sources):
                prediction_ = [predictions[i]]
                reference_ = [references[i]]
                score = self.metrics[method](prediction_, reference_)[-1]
                score_ = round(score, float_)
                case_score.append(score_)
        return case_score

    '''
    General evaluation
    '''
    def general_evaluation(self, method, float_, sources=None, predictions=None, references=None):
        if method == 'COMET':
            refs = len(references[0])
            ref_all = list()
            scores = list()
            for ref_num in range(refs):
                ref_ = [item[ref_num] for item in references]
                ref_all.append(ref_)
            for ref_detail in ref_all:
                score = self.metrics[method](sources, predictions, ref_detail)[-1]
                score = round(score, float_)
                scores.append(score)
            return sum(scores)/len(scores)
        else:
            score = self.metrics[method](sources, predictions, references)[-1]
            score = round(score, float_)
            return score











