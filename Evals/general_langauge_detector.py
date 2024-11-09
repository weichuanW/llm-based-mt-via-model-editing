import re

from polyglot.detect import Detector
from langdetect import detect_langs
import gcld3
import fasttext

import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector


class LanguageDetectors(object):
    def __init__(self):
        self.detectors_ = {'polyglot': self.detector_polyglot,
                           'langdetect': self.detector_langdetect,
                           'gcld3': self.detector_gcld3,
                           'fasttext': self.detector_fasttext,
                           }
        # 'gcld3': self.detector_gcld3,
        # 'fasttext': self.detector_fasttext,
        # 'spacy_s': self.detector_spacy_single, 1.55s/item
        # 'spacy_m': self.detector_spacy_multiple

    '''
    D: language detector with polyglot
    I: detect content
    O: detect language [str] and related confidence [float(default with.4)]
    E:
    '''

    def detector_polyglot(self, text):
        try:
            if len(text) < 1:
                return 'no', 0.0
            detector = Detector(text)
            detected_language = detector.language.code
            confidence = detector.language.confidence
            return detected_language, round(confidence, 4)
        except Exception as e:
            print("polyglot: This row throws and error:", text)
            print('This is the text {}. The details'.format(text))
            print("An error occurred:", e)
            return None, 0.0

    '''
    D: language detector with langdetect
    I: detect content
    O: detect language [str] and related confidence [float(default with.4)]
    E:
    *N: handle the language zh-ch to zh
    '''

    def detector_langdetect(self, text):
        try:
            if len(text) < 1:
                return 'no', 0.0
            langs = {lang.lang: lang.prob for lang in detect_langs(text)}
            language = list(langs.keys())[0]
            language_score = list(langs.values())[0]
            if language == 'zh-cn':
                language = 'zh'
            return language, round(language_score, 4)
        except Exception as e:
            print("langdetect: This row throws and error:", text)
            print('This is the text {}. The details'.format(text))
            print("An error occurred:", e)
            return None

    '''
    D: language detector with gcld3
    I: detect content
    O: detect language [str] and related confidence [float(default with.4)]
    E:
    '''

    def detector_gcld3(self, text):
        try:
            if len(text) < 1:
                return 'no', 0.0
            detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
            result = detector.FindLanguage(text=text)
            # print(result.probability, result.proportion)
            lang_detected, score = result.language, result.probability
            return lang_detected, round(score, 4)
        except Exception as e:
            print("gcld3: This row throws and error:", text)
            print('This is the text {}. The details'.format(text))
            print("An error occurred:", e)
            return None

    '''
    D: language detector with fasttext
    I: detect content
    O: detect language [str] and related confidence [float(default with.4)]
    E:
    '''

    def detector_fasttext(self, text):
        try:
            if len(text) < 1:
                return 'no', 0.0
            model = fasttext.load_model("/home/weichuanw/95server/WorkFlow_v3.0/Model/models/lid.176.bin")
            lang_detected = model.predict(text)
            score = round(lang_detected[1][0])
            # print(lang_detected)
            return lang_detected[0][0].replace('__label__', ''), round(score, 4)
        except Exception as e:
            print("fasttext: This row throws and error:", text)
            print('This is the text {}. The details'.format(text))
            print("An error occurred:", e)
            return None

    '''
    D: load language detector for spacy
    I: ..
    '''
    def get_lang_detector(self, nlp, name):
        return LanguageDetector(seed=42)

    '''
    D: language detector with spacy for document level detection (regard the input as single instance)
    I: detect content
    O: detect language [str] and related confidence [float(default with.4)]
    E:
    '''
    def detector_spacy_single(self, text):
        try:
            nlp_model = spacy.load('en_core_web_lg')
            Language.factory("language_detector", func=self.get_lang_detector)
            nlp_model.add_pipe('language_detector', last=True)

            doc = nlp_model(text)
            language = doc._.language
            return language['language'], round(language['score'], 4)
        except Exception as e:
            print("This row throws and error:", text)
            print('This is the text {}. The details'.format(text))
            print("An error occurred:", e)
            return None

    '''
    D: language detector with spacy for multiple sentence level detection (regard the input as multiple instances)
    I: detect content
    O: sentence instance with corresponding language and scores [dict[dict[]]]
    E:
    '''
    def detector_spacy_multiple(self, text):
        try:
            # nlp_model = spacy.load('en_core_web_lg')
            nlp_model = spacy.load("en_core_web_sm")
            Language.factory("language_detector", func=self.get_lang_detector)
            nlp_model.add_pipe('language_detector', last=True)

            doc = nlp_model(text)
            all_output = dict()
            for i, sent in enumerate(doc.sents):
                all_output[sent] = sent._.language
            return all_output
        except Exception as e:
            print("This row throws and error:", text)
            print('This is the text {}. The details'.format(text))
            print("An error occurred:", e)
            return None

