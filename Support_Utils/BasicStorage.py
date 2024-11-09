# python>=3.7.0
# store data for various files
# store data with input file
# class: BasicStorage

from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import docx2txt
from xml.etree import cElementTree as ET
from translate.storage.tmx import tmxfile
import csv
import torch

class BasicStorage(object):
    '''
    D: This class is only for basic data storage, so there are only two types for __init__.
    -- The basic parameters for file name and file type. The basic handling operation is judging
    -- the file type with suffix and return the data with default type
    I: data, file name [str] file type [str] (which is actually the file suffix)
    O: Storage results and storage path [str] [str]
    E: I: [1, 2], 'a.txt', 'txt' O: 'Storage finished'. '/storage/path/a.txt'
    *N: We state that many data can be stored in a file with flexible suffix, but for the consistency of the whole code frame
    --we still do the file type check. However, we provide a Basic storage just storage data with a line-by-line way
    --usage: BS(data, file_name).data_reader[file_type]()
    '''

    def __init__(self, data, file_name):
        self.file_name = file_name
        self.data = data
        self.data_reader = {'txt': self.txt_storage,
                            'json': self.json_storage,
                            'csv': self.csv_storage,
                            'docx': self.docx_storage,
                            'npy': self.npy_storage,
                            'pt': self.pt_storage
                            }
        # we delete the above line because it has a conflict with our original design
        # self.data = self.data_reader[self.file_type]()

    '''
    D: this is a template for quick starting
    I: 
    O: 
    E: 
    '''
    def template(self):
        pass

    '''
    D: this is a template for quick starting
    I: file name [str] and data [list] from init function
    O: finished description and total length (lines)
    E: .. 
    '''
    def txt_storage(self):
        with open(self.file_name, 'w', encoding='utf-8') as fin:
            for i, item in enumerate(self.data):
                if i == len(self.data) - 1:
                    string = item.strip()
                    fin.write(string)
                else:
                    string = item.strip() + '\n'
                    fin.write(string)
        print('The storage of {} has been finished with {} lines'.format(self.file_name, len(self.data)))


    '''
    D: this is a storage function for json file with dict type
    I: file name [str] and data [dict] from init function
    O: finished description and file name (lines)
    E: ..
    '''
    def json_storage(self):
        with open(self.file_name, 'w+', encoding='utf-8') as fin:
            json.dump(self.data, fin)
        print('The storage of {} has been finished with {} elements'.format(self.file_name, len(self.data)))


    '''
    D: storage csv data to a csv file
    I: pandas data with pandasformat [pdformat]
    O: storage notification
    E: 
    '''
    def csv_storage(self):
        self.data.to_csv(self.file_name, encoding='utf-8', index=False)
        print('The storage of {} has been finished with {} elements'.format(self.file_name, len(self.data)))

    '''
    D: this is a template for quick starting
    I: 
    O: 
    E: 
    '''
    def docx_storage(self):
        pass


    '''
    D: this is a storage function for npy file with np.array type
    I: file name [str], data [np.array]
    O: finished description with file name and data size
    E: 
    '''
    def npy_storage(self):
        with open(self.file_name, 'wb') as fin:
            np.save(fin, self.data)
        print('The storage of {} has been finished with {} elements'.format(self.file_name, self.data.shape))

    '''
    D: this is a storage function for pt file with torch.tensor type
    I: file name [str], data [torch.tensor]
    O: finished description with file name and data size
    E:
    '''
    def pt_storage(self):
        torch.save(self.data, self.file_name)
        print('The storage of {} has been finished with {} elements'.format(self.file_name, self.data.shape))

