import os

# from the Investigating the Translation Performance of a Large Multilingual Language Model: the Case of BLOOM
# template table 1

class Prompt_Template_Sum(object):
    def __init__(self, type):
        self.type = type
        if type == 'zero':
            self.templates = {'temp1': self.tab1_zero_temp1,
                              'temp2': self.tab1_zero_temp2,
                              'temp3': self.tab1_zero_temp3,
                              'temp4': self.tab1_zero_temp4,
                              'temp5': self.tab1_zero_temp5,
                              'temp6': self.tab1_zero_temp6}
        elif type == 'one':
            self.templates = {'temp1': self.tab1_one_temp1,
                              'temp2': self.tab1_one_temp2,
                              'temp3': self.tab1_one_temp3,
                              'temp4': self.tab1_one_temp4,
                              'temp5': self.tab1_one_temp5}
        else:
            raise ValueError('The type is not supported')

    def tab1_zero_temp1(self, src_lang, trg_lang, src_content):
        format_ = f'Given the following source text: {src_content}, a good {trg_lang} translation is:'
        return format_

    def tab1_zero_temp2(self, src_lang, trg_lang, src_content):
        format_ = f'If the original version says {src_content} then the {trg_lang} version should say:'
        return format_

    def tab1_zero_temp3(self, src_lang, trg_lang, src_content):
        format_ = f'What is the {trg_lang} translation of the sentence: {src_content}?'
        return format_

    def tab1_zero_temp4(self, src_lang, trg_lang, src_content):
        format_ = f'{src_lang}:{src_content} = {trg_lang}:'
        return format_

    def tab1_zero_temp5(self, src_lang, trg_lang, src_content):
        format_ = f'{src_content} translates into {trg_lang} as:'
        return format_

    def tab1_zero_temp6(self, src_lang, trg_lang, src_content):
        format_ = f'{src_lang}:{src_content}\n{trg_lang}:'
        return format_

    def tab1_one_temp1(self, exp_src, exp_trg, src_lang, trg_lang, src_content):
        exp_template = f'Given the following source text: {exp_src}, a good {trg_lang} translation is: {exp_trg}\n'
        source_template = f'Given the following source text: {src_content}, a good {trg_lang} translation is:'
        return exp_template+source_template

    def tab1_one_temp2(self, exp_src, exp_trg, src_lang, trg_lang, src_content):
        exp_template = f'If the original version says {exp_src} then the {trg_lang} version should say: {exp_trg}\n'
        source_template = f'If the original version says {src_content} then the {trg_lang} version should say:'
        return exp_template+source_template

    def tab1_one_temp3(self, exp_src, exp_trg, src_lang, trg_lang, src_content):
        exp_template = f'What is the {trg_lang} translation of the sentence: {exp_src}? {exp_trg}\n'
        source_template = f'What is the {trg_lang} translation of the sentence: {src_content}?'
        return exp_template+source_template

    def tab1_one_temp4(self, exp_src, exp_trg, src_lang, trg_lang, src_content):
        exp_template = f'{src_lang}:{exp_src} = {trg_lang}:{exp_trg}\n'
        source_template = f'{src_lang}:{src_content} ='
        return exp_template+source_template

    def tab1_one_temp5(self, exp_src, exp_trg, src_lang, trg_lang, src_content):
        exp_template = f'{exp_src} translates into {trg_lang} as: {exp_trg}\n'
        source_template = f'{src_content} translates into {trg_lang} as:'
        return exp_template+source_template



# from the paper: Steering Large Language Models for Machine Translation with Finetuning and In-Context Learning
# template table 3
class Prompt_Instruct_template(object):
    def __init__(self, type):
        if type == 'zero':
            self.templates = {'instruct1': self.instruct_zero}
        elif type == 'one':
            self.templates = {'instruct1': self.instruct_one_1,
                              'instruct2': self.instruct_one_2,
                              'instruct3': self.instruct_one_3}
        else:
            raise ValueError('The type is not supported')
    def instruct_zero(self, src_lang, trg_lang, src_content):
        format_ = f'Translate the source text from {src_lang} to {trg_lang}.\nSource: {src_content} Target:'
        return format_
    def instruct_one_1(self, exp_src, exp_trg, src_lang, trg_lang, src_content):
        exp_template = f'Translate the source text from {src_lang} to {trg_lang}.\nSource: {exp_src}\nTarget: {exp_trg}\n'
        source_template = f'Translate the source text from {src_lang} to {trg_lang}.\nSource: {src_content}\nTarget:'
        return exp_template+source_template

    def instruct_one_2(self, exp_src, exp_trg, src_lang, trg_lang, src_content):
        exp_template = f'Consider the following 2 translations from {src_lang} to {trg_lang}.\nExample 1\nSource: {exp_src}\nTarget: {exp_trg}\n'
        source_template = f'Translate the source text from {src_lang} to {trg_lang}.\nSource: {src_content}\nTarget:'
        return exp_template+source_template

    def instruct_one_3(self, exp_src, exp_trg, src_lang, trg_lang, src_content):
        exp_template = f'Consider the following translations from {src_lang} to {trg_lang}.\nSource: {exp_src}\nTarget: {exp_trg}\n'
        source_template = f'Translate the source text from {src_lang} to {trg_lang}.\nSource: {src_content}\nTarget:'
        return exp_template+source_template

'''# testing
src_lang = 'English'
trg_lang = 'Chinese'
src_content = 'Hello, where are you?'
exp_src = 'Hello, how are you?'
exp_trg = '你好，你好吗？'

# testing on temp1
temps = Prompt_Template_Sum('zero')
print('template1 \n')
print(temps.templates['temp1'](src_lang, trg_lang, src_content))
print('template2 \n')
print(temps.templates['temp2'](src_lang, trg_lang, src_content))
print('template3 \n')
print(temps.templates['temp3'](src_lang, trg_lang, src_content))
print('template4 \n')
print(temps.templates['temp4'](src_lang, trg_lang, src_content))
print('template5 \n')
print(temps.templates['temp5'](src_lang, trg_lang, src_content))

temps = Prompt_Template_Sum('one')
print('template1 for one-shot\n')
print(temps.templates['temp1'](exp_src, exp_trg, src_lang, trg_lang, src_content))
print('template2 for one-shot\n')
print(temps.templates['temp2'](exp_src, exp_trg, src_lang, trg_lang, src_content))
print('template3 for one-shot\n')
print(temps.templates['temp3'](exp_src, exp_trg, src_lang, trg_lang, src_content))
print('template4 for one-shot\n')
print(temps.templates['temp4'](exp_src, exp_trg, src_lang, trg_lang, src_content))
print('template5 for one-shot\n')
print(temps.templates['temp5'](exp_src, exp_trg, src_lang, trg_lang, src_content))

# testing for instruction
instruct = Prompt_Instruct_template('zero')
print('instruction 1\n')
print(instruct.templates['instruct1'](src_lang, trg_lang, src_content))
instruct = Prompt_Instruct_template('one')
print('instruction 1 on one-shot \n')
print(instruct.templates['instruct1'](exp_src, exp_trg, src_lang, trg_lang, src_content))
print('instruction 2 on one-shot \n')
print(instruct.templates['instruct2'](exp_src, exp_trg, src_lang, trg_lang, src_content))
print('instruction 3 on one-shot \n')
print(instruct.templates['instruct3'](exp_src, exp_trg, src_lang, trg_lang, src_content))'''