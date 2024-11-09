# python>=3.7.0
# read data from various files
# read the dataset path, DataLoadScripts and basic dataset information
# class: BasicReading

from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import docx2txt
from xml.etree import cElementTree as ET
from translate.storage.tmx import tmxfile
from .BasicFiltering import BasicFiltering
import torch

class BasicReading(object):
    '''
    D: This class is only for basic data reading, so there are only two types for __init__.
    -- The basic parameters for file name and file type. The basic handling operation is judging
    -- the file type with suffix and return the data with default type
    I: file name [str]
    E: I: 'a.txt', 'txt' O by using object.data_reader[object.file_type](sep=''(optional)) get [a, b, c, d]
    *N: we provide a Basic_reading just read file line-by-line without any other operation
    --basic usage: BR(file_name).data_reader[file_type]()
    '''
    def __init__(self, file_name):
        self.file_name = file_name
        #this part only store the function location so it is ok
        self.data_reader = {'txt': self.txt_reading,
                            'json': self.json_reading,
                            'csv': self.csv_reading,
                            'docx': self.docx_reading,
                            'xml': self.xml_reading,
                            'tmx': self.tmx_reading,
                            'npy': self.npy_reading,
                            'pt': self.pt_reading,
                            }

    '''
    D: this is a template for quick starting
    I: 
    O: 
    E: 
    '''
    def template(self):
        pass

    '''
    D: reading npy file 
    I: default self.file_name
    O: output the matrix version of data [ndarray] 
    --(note that this format can include different data type, but must be the same)
    E: I: , O: [2,3,5,2](dtype=int)
    '''
    def npy_reading(self):
        with open(self.file_name, 'rb') as fin:
            data = np.load(fin)
        return data

    '''
    D: reading tmx file (this is a general file for NLP area)
    I: default self.file_name, source language, target language (the abbreviation)
    O: content list for source language [list] and target language[list]
    E: I: 'en', 'de', O: [xxx, xxx], [xxx, xxx]
    '''
    def tmx_reading(self, source_lang, target_lang):
        if not source_lang and not target_lang:
            with open(self.file_name, 'rb') as fin:
                content = tmxfile(fin, source_lang, target_lang)
            src = list()
            trg = list()
            for node in content.unit_iter():
                src.append(node.source)
                trg.append(node.target)
        else:
            raise ValueError('Please input extract elements')
        return src, trg

    '''
    D: reading xml file (this is a general file for NLP area)
    I: default self.file_name, extract elements [str]
    O: content list for specific element [str]
    E: I: , 'tags'
    '''
    def xml_reading(self, extract_element=None):
        tree = ET.parse(self.file_name)
        root = tree.getroot()
        context = list()
        if extract_element:
            for seg in tqdm(root.iter(extract_element)):
                context.append(seg.text)
        else:
            raise ValueError('Please input extract elements')
        return context

    '''
    D: reading docx files (word 07-now) and output list, default is reading content line by line
    I: default by self.file_name, separation sequence [str] (optional), default is '\n\n'
    O: list content [list]
    E: I: , '\n' O: [a,b,c]
    *N: we recommend you to use customised separation by calling this function uniquely
    -- TODO: figure out the detailed reading content
    '''
    def docx_reading(self, sep=None):
        all_text =docx2txt.process(self.file_name)
        if not sep:
            data = all_text.strip().split('\n')
        else:
            data = all_text.strip().split(sep)
        content = BasicFiltering().list_remove_null(data)
        return content

    '''
    D: reading csv file and output pandas.DataFrame
    I: default by self.file_name, separation signals (optional) [str]
    O: a Pandas.DataFrame [DataFrame]
    E: I: 'a.csv' O: DataFrame({a:[1,2],b:[2,3],...})
    *N: if you want to read the csv file, you can directly call this function
    -- rather than object.data with specific seq
    '''
    def csv_reading(self, sep=None):
        if sep:
            data = pd.read_csv(self.file_name, sep=sep)
        else:
            data = pd.read_csv(self.file_name)
        return data

    '''
    D: reading json file and output dictionary
    I: default by self.file_name
    O: content by dictionary [dict]
    E: I: O: {a:1,b:2,...}
    '''
    def json_reading(self):
        with open(self.file_name, 'r+', encoding='utf-8') as fin:
            data = json.load(fin)
        return data

    '''
    D: reading text file line by line and split content line by line
    I: default from self.filename
    O: list without any empty content [list]
    E: I: O: [a, b, c, d]
    '''
    def txt_reading(self):
        with open(self.file_name, 'r+', encoding='utf-8') as fin:
            text = fin.read()
            content = text.split('\n')
        content = BasicFiltering().list_remove_null(content)
        return content

    '''
    D: reading pt file and output the content
    I: default by self.file_name
    O: content by torch.load [torch.tensor]
    E: I: 'a.pt' O: tensor([1,2,3,4])
    '''
    def pt_reading(self):
        return torch.load(self.file_name)

    '''
    D: reading any file line by line and maintain all content include empty lines
    I: default from self.filename
    O: list with all content [list] (including empty line)
    E: I: O: [a, b, c, d]
    '''
    def Basic_reading(self):
        with open(self.file_name, 'r+', encoding='utf-8') as fin:
            text = fin.read()
            content = text.split('\n')
        return content






