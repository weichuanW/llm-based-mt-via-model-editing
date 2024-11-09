from collections import Counter
import string as ST
import zhon.hanzi
from collections import OrderedDict


# finished: support for multiple languages
class RepeRetrieval(object):

    '''
    D: initialisation for the punctuation
    '''
    def __init__(self):
        self.Chinese_punctuation = ST.punctuation + zhon.hanzi.punctuation

    '''
    D: count words and corresponding frequency in a list
    I: list [list]
    O: count dictionary [dict](key:value = word:frequency)
    E:
    -- I: ['a', 'b', 'c', 'a', 'b', 'a']
    -- O: {'a': 3, 'b': 2, 'c': 1}
    '''
    def count_frequency_list(self, list_):
        string_counts = Counter(list_)
        result_dict = dict(string_counts)
        # remove the value with 1 frequency
        result_dict_remove = {key: value for key, value in result_dict.items() if value > 1}
        return result_dict_remove

    '''
    D: divide string into a list of words based on language and count words and corresponding frequency
    I: string [str], language [str](default is 'zh')
    O: count dictionary [dict](key:value = word:frequency)
    E:
    -- I: '我爱你'
    -- O: {'我': 1, '爱': 1, '你': 1}
    '''
    def count_initial_words(self, string_, lang='zh'):
        initial_words = list()
        if lang == 'zh':
            for item in string_:
                initial_words.append(item)
        else:
            items = string_.split(' ')
            for item in items:
                initial_words.append(item)
        # remove the 1 frequency words

        return self.count_frequency_list(initial_words)

    '''
    D: first check the repetition with a initial threshold of 8
    I: count dictionary [dict], threshould [int](default is 8)
    O: True for existing repetition, False for not repetition
    E:
    -- I: {'a': 3, 'b': 2, 'c': 1}, threshould = 8
    -- O: False
    '''
    def check_repe_threshould(self, count_dict, threshould=8):
        count_dict_ = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        if count_dict_[0][1] < threshould + 1:
            return False
        else:
            return True

    '''
    D: filter the word with the predefined threshold for following repetition retrieval
    I: count dictionary [dict], threshould [int](default is 8)
    O: filtered count dictionary [dict](key:value = word:frequency)
    E:
    -- I: {'a': 3, 'b': 2, 'c': 1}, threshould = 2
    -- O: {'a': 3}
    '''
    def filter_repe_threshold(self, count_dict, threshould=8):
        filtered_dict = {key: value for key, value in count_dict.items() if value >= threshould}
        return filtered_dict

    '''
    D: check whether all repetition is punctuation
    I: count dictionary [dict]
    O: True for all punctuation, False for not all punctuation
    *N if a repetition all things are punctuations, then we should ensure the repetitions are consistent punctuations
    -- e.g. 我是！！！！！！is illegal, while 我是！！！好人！！！is legal
    '''
    def check_punct(self, count_dict):
        new_count = dict()
        for item in count_dict:
            if item in self.Chinese_punctuation:
                pass
            else:
                new_count[item] = count_dict[item]
        if len(new_count) != 0:
            return False
        else:
            return True

    '''
    D: filter the unique characters in a string (find the substring in the repetition)
    I: string [str]
    O: filtered string [str]
    '''
    def filter_unique_chars_ordered(self, string_):
        return ''.join(OrderedDict.fromkeys(string_))

    '''
    D: repetition retrieval for the initial repetition and the end repetition
    I: initial words [str], filtered dict [dict]
    O: first repetition happened position, second repetition position, 
    -- end repetition position (general the last position, while special for the failed decoding)
    -- connection punctuations
    *N: all repetition should be consistent and can be all punctuations
    '''
    def repetition_position(self, initial_words, filtered_dict):
        # stage 1, storage the initial ids ordering for potential repetition
        temp_ids = list()
        connect_punc = list()

        # ordering storage
        for i, item in enumerate(initial_words):
            if item in filtered_dict:
                temp_ids.append(i)
            else:
                pass
        # stage 2, find the first and last repetition position
        begin_id = -1
        end_id = -1
        for i, item in enumerate(temp_ids):
            # do not need to check the last position
            if i == len(temp_ids) - 1:
                pass
            else:
                if begin_id == -1:
                    # check whether all the characters are punctuation
                    if initial_words[item] in self.Chinese_punctuation and self.check_punct(filtered_dict):
                        begin_id = item
                        # print(initial_words[item])
                    # if the characters are not all punctuations, then we need to check the next character
                    elif initial_words[item] in self.Chinese_punctuation:
                        begin_id = temp_ids[i + 1]
                    else:
                        begin_id = item
                    # print(begin_id)
                else:
                    # consistency check for the next character in repetition
                    if item + 1 in temp_ids:
                        end_id = item + 1
                    # addtional connection punctuations
                    elif initial_words[item + 1] in self.Chinese_punctuation:
                        if initial_words[item + 1] not in connect_punc:
                            connect_punc.append(initial_words[item + 1])
                        end_id = item + 1
                    else:
                        # print(initial_words[item + 1])
                        # initial searching
                        # one case is 我爱我家我的我的我的我的。。。 the beginning id should be 4 rather than 0 or 2
                        begin_id = -1
                        end_id = -1
        # find the second repetition position, equal to find the same substring in the repetition area
        if begin_id == -1:
            return -1, -1, -1, connect_punc
        repetition_string = initial_words[begin_id:end_id + 1]
        subrepe = self.filter_unique_chars_ordered(repetition_string)
        subrepe_list = list(subrepe)
        # for each character in substring, it must hit one of the frequency words, so we remove the character until
        # the substring is empty
        gap = 0
        second_id = -1
        for i, item in enumerate(repetition_string):
            if item in filtered_dict:
                subrepe_list.remove(item)
            else:
                pass
            if len(subrepe_list) == 0:
                gap = i + 1
                break
        if gap == 0:
            pass
        else:
            second_id = begin_id + gap


        return begin_id, second_id, end_id, connect_punc


    def repetition_retrieval_record(self, initial_string, lang='zh'):
        # get the frequency dict
        initial_count = self.count_initial_words(initial_string, lang=lang)
        # get the word piece string with the list type
        # e.g. I have a pen -> ['I', ' ', 'have', ' ', 'a', ' ', 'pen']
        word_piece = initial_string.split(' ') if lang != 'zh' else list(initial_string)

        # the last one may be special to include the punctuation repetitions
        word_last = word_piece[-1]
        # remove the last one since they may miss the generation
        word_piece = word_piece[:-1]
        # initial the searching tools
        whole_pointer = 0
        substring_gap = 0
        substring_pointer = 0
        find_flag = False
        for i in range(len(word_piece)):
            # if find the substring, break
            if find_flag:
                break

            # save searching time
            if i < whole_pointer:
                continue

            # if the character is not in the initial count, then we need to check the next character
            if i == len(word_piece) - 1:
                break

            if word_piece[i] not in initial_count:
                continue

            else:
                substring_gap = 0
                substring_pointer = i
                substring_beginner = word_piece[i]
                # record for temp flag finding the substring

                # stage 1: build the repetition string
                # error raise here, some words does not finish the generation
                for j in range(1, len(word_piece)-i):
                    if word_piece[i+j] not in initial_count:
                        whole_pointer = i + j
                        break
                    else:

                        if word_piece[i+j] == substring_beginner:
                            substring_gap = j
                            break
                # stage 2: check the repetition string pattern for the whole string
                # the issue is here
                if substring_gap != 0:
                    temp_flag = True
                    max_len = (len(word_piece) - substring_pointer) // substring_gap
                    new_max = min(len(word_piece), substring_pointer + substring_gap * max_len)
                    for q in range(substring_pointer, new_max, substring_gap):
                        # skip the final position
                        if q == len(word_piece) - 1:
                            break

                        if word_piece[q] != substring_beginner:
                            temp_flag = False
                            break
                        else:
                            pass
                    find_flag = temp_flag
                else:
                    pass
        # check for number or punctuation repetitions
        # check for the last position
        if lang!= 'zh' and not find_flag:
            # for the punctuation, we think the repetition within three is normal representation
            if len(word_last) > 3:
                whole_pointer, substring_pointer, substring_gap = self.repetition_retrieval_record(word_last, lang='zh')
                whole_pointer = -1
        return whole_pointer, substring_pointer, substring_gap

    '''
    D: second check the repetition phenomenon with the position checking
    I: repetition position [tuple]
    O: True for repetition, False for not repetition
    *N:  for the first three elements, if the value is -1, then it means the continual repetition is not exist
    -- for the last element, if the length is larger than 3, then it means the connection punctuations are too many,
    -- we should reconsider the repetition
    '''
    def check_repe_continual(self, tuple_):
        if tuple_[0] == -1:
            return False
        elif tuple_[1] == -1:
            return False
        elif tuple_[2] == -1:
            return False
        elif len(tuple_[3]) > 3:
            return False
        else:
            return True


    '''
    D: repetition retrieval for the a list with multiple strings
    I: judgement list [list[str]], language [str](default is 'zh'), threshould [int](default is 8), 
    -- default generation setting on the token to filter some punctuation only repetitions
    O: repetition content [list[tuple[int, str, tuple(int, int, int)]]], repetition ids [list[int]]
    E:
    -- O: [(1, 'I like it it it it it', (3, 4,7))]
    '''
    def repetition_retrieval(self, judge_repe, lang='zh', threshould=8, token_setting=200):

        repetition_all = list()
        repetition_ids = list()

        for i, check_string in enumerate(judge_repe):
             # initial judgement on length, the repeated generation must finish the generation with full token setting
            if len(check_string) <= int(token_setting // 3):
                continue
            # initial count
            initial_count = self.count_initial_words(check_string, lang=lang)
            # initial check
            check_initial = self.check_repe_threshould(initial_count, threshould=threshould)
            if check_initial:
                # filter the initial count
                filter_count = self.filter_repe_threshold(initial_count, threshould=threshould)
                # repetition position
                repetition_pos = self.repetition_position(check_string, filter_count)
                # check the repetition
                check_repe = self.check_repe_continual(repetition_pos)
                if check_repe:
                    # storage for the repetition id, repetition string, repetition position tuple (first, second, end)
                    storage_ = (i, check_string, repetition_pos[:-1])
                    repetition_ids.append(i)
                    repetition_all.append(storage_)
                else:
                    pass
        return repetition_all, repetition_ids
    '''
    D: repetition retrieval for the a list with multiple strings
    I: judgement list [list[str]], tokenizer [transformers.tokenizer], language [str](default is 'zh', or 'en'), token_setting = 400
    -- please remember that we will use the zh to represent a kind of languages with without space to break the words
    O: pure repetition ids [list[int]]
    -- repetition content with detailed information for repetition id, repetition beginning words, repetition gap [list[tuple[int, int, int]]], 
    -- repetition string [list[tuple[int, str, str]]] for convenient applying
    -- exceptions for the repetition [list[str]]
    please remember that the position is for the word position rather than the case position, please concatenate the string to get the real position
    '''
    def repetition_judgement_retrieval(self, judge_repe, tokenizer, lang='zh', token_setting=400):
        # only recording for repetition ids
        repetition_ids = list()
        # recording the sub repetition with the format of [(id, repetition start position, repetition gap)]
        repetition_retriever = list()
        # recording for repetition str representation
        repetition_str = list()
        # token counting
        token_str = list()

        excep = list()
        for i in range(len(judge_repe)):
            token_str.append(tokenizer(judge_repe[i], return_tensors="pt", max_length=1024)["input_ids"].shape[1])
        for i, check_string in enumerate(judge_repe):
            # initial judgement on token length
            # pass the non-repetitions
            # these non-repetition will not be recorded
            # special design for Chinese and Japanese with multiple token encoding design
            '''if '�' in check_string:
                if check_string.count('�') >= 2:
                    #sub_token = check_string.count('�') // 2
                    #token_setting = token_setting - sub_token
                    token_setting=1'''
            if token_str[i] < token_setting:
                continue
            # retrieval for sub repetition
            whole_pointer, substring_pointer, substring_gap = self.repetition_retrieval_record(check_string, lang=lang)
            if substring_gap != 0:
                repetition_ids.append(i)
                # when the whole pointer is positive, then we need to reconstruct the repetition on the normal string
                # if the whole pointer is negative, then we need to reconstruct the repetition on the last string
                if whole_pointer != -1:
                    repetition_retriever.append((i, 0, (substring_pointer, substring_gap)))
                    original_without_str, original_repe_str = self.reconstruct_repes(check_string, substring_pointer,
                                                                                     substring_gap, lang=lang)
                else:
                    repetition_retriever.append((i, -1, (substring_pointer, substring_gap)))
                    original_without_str, original_repe_str = self.reconstruct_repes(check_string, substring_pointer,
                                                                                     substring_gap, lang=lang, last=True)

                repetition_str.append((i, original_without_str, original_repe_str))
            else:
                excep.append(check_string)
                #print(check_string)
        return repetition_ids, repetition_retriever, repetition_str, excep

    '''
    D: reconstruct the repetition substrings based on our detected indices and gaps
    '''
    def reconstruct_repes(self, initial_string, beginner, gap, lang='zh', last=False):
        if not last:
            word_piece = initial_string.split(' ') if lang != 'zh' else list(initial_string)
            without_repe = word_piece[:beginner]
            original_without_str = ' '.join(without_repe) if lang != 'zh' else ''.join(without_repe)
            # the repetition string
            repetition_string = word_piece[beginner:beginner + gap]
            original_repe_str = ' '.join(repetition_string) if lang != 'zh' else ''.join(repetition_string)
        else:
            word_piece = initial_string.split(' ') if lang != 'zh' else list(initial_string)
            without_repe = word_piece[:-1]
            last_element = list(word_piece[-1])

            original_without_str = ' '.join(without_repe) if lang != 'zh' else ''.join(without_repe)
            original_without_str = original_without_str + ' ' + ''.join(last_element[:beginner])
            # the repetition string
            repetition_string = last_element[beginner:beginner + gap]
            original_repe_str = ''.join(repetition_string)
        return original_without_str, original_repe_str

# test for repetition
'''import base_load as bl
data = bl.BR('../../temp_local_files/temp_local_2_5/excep_repes.txt').data_reader['txt']()
data_test = data[-1]
test = RepeRetrieval()
data_split = data_test.split(' ')
print(test.repetition_retrieval_record(data_test, lang='en'))'''

# another issue for the punctuation repetition in the last position
