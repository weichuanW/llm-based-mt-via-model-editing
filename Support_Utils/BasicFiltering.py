# doing data filtering, including data fusion, data filtering and data counting
import string as ST
import zhon.hanzi
import regex as re

class BasicFiltering(object):
    '''
    D: provide basic filtering with language-agnostic (space) and languages-specific (special punctuation), support for
    --support for sequence with string and int type
    E:
    '''
    def __init__(self):
        pass


    '''
    D: this is a template for quick starting
    I: 
    O: 
    E: 
    '''
    def template(self):
        pass

    '''
   D: remove special characters while stay the number, alphabelt and space
   I: sequence string [str]
   O: filtered string without special characters
   E: I: #$% Have a fr11ewan!() && eat Haveafr11ewaneat
      O: Have a fr11ewan  eat
   '''
    def string_removespecial(self, sequence):
        item = re.sub(r'[^a-zA-Z0-9\s]+', '', item)

    '''
    D: remove space within a sequence
    I: a string [str]
    O: a string without space [str]
    E: I: 'I have a dream', O: 'Ihaveadream'
    '''
    def string_removespace(self, sequence):
        return sequence.replace(' ', '')

    '''
    D: remove general punctuations within a sequence (Default by the English version, fastest)
    I: a string [str]
    O: a string without space [str]
    E: I: 'I have, a dream', O: 'I have a dream'
    '''
    def string_removepunc(self, sequence):
        return sequence.translate(str.maketrans('', '', ST.punctuation))
    #todo: add description
    '''
    D
    '''
    def string_removedigit(self, sequence):
        pattern = r'[0-9]'
        # Match all digits in the string and replace them with an empty string
        new_string = re.sub(pattern, '', sequence)
        return new_string

    '''
    D: remove Unicode punctuations within a sequence
    I: a string [str]
    O: a string without space [str]
    E: I: 'I have, a dream', O: 'I have a dream'
    '''
    def string_removepunc_U(self, sequence):
        return re.sub(u"\p{P}+", "", sequence)

    '''
    D: remove Chinese punctuations
    I: a string [str]
    O: a string without space [str]
    E: I: 'I have a dream', O: 'Ihaveadream'
    '''
    def string_removepunc_zh(self, sequence):
        return sequence.translate(str.maketrans('', '', zhon.hanzi.punctuation))

    '''
    D: remove deduplication in list (str only)
    I: a list [list]
    O: a index list without deduplication (original list index) [list]
    E: I: [1,2,1], O: [1,2]
    '''
    def list_deduplication_string(self, sequence_list):
        reconstruct = list()
        filter_index = list()
        for i, item in enumerate(sequence_list):
            filter_item = self.string_removepunc_U(item)
            filter_item = self.string_removespace(filter_item)
            if filter_item not in reconstruct:
                reconstruct.append(filter_item)
                filter_index.append(i)

        return filter_index

    '''
    D: remove deduplication in list (str only)
    I: a list [list]
    O: a index list without deduplication (original list index) [list]
    E: I: [1,2,1], O: [0,1] (index)
    '''
    def list_deduplication_common_order(self, list_):
        reconstruct = list()
        filter_index = list()
        for i, item in enumerate(list_):
            if item not in reconstruct:
                reconstruct.append(item)
                filter_index.append(i)

        return filter_index


    '''
    D: remove deduplication in list (str only)
    I: a list [list]
    O: new list without deduplication [list]
    E: I: [1,2,1], O: [1,2]
    '''
    def list_deduplication_common_list(self, list_):
        return list(set(list_))


    '''
    D: filtering the empty elements in a list
    I: filtering list [list]
    O: new content list without empty elements [list]
    E: I for [a, b, c, ''] O for [a, b, c] (a, b, c are data)
    '''
    def list_remove_null(self, list_):
        new_content = list()
        for item in list_:
            if len(item) < 1:
                pass
            else:
                new_content.append(item)
        return new_content
