from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
from transformers import StoppingCriteriaList, StoppingCriteria
from tqdm import tqdm
import torch.nn.functional as F

class KeywordsStoppingCriteriaCase(StoppingCriteria):
    '''
    D: designed for early stopping with pre-defined keywords on case inference
    I: keywords ids [list]
    '''
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    '''
    D: designed for early stopping with pre-defined keywords
    '''
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # only need to detect the last token (new generated)
        if input_ids[0][-1] in self.keywords:
            return True
        return False

class KeywordsStoppingCriteriaBatch(StoppingCriteria):
    '''
    D: designed for early stopping with pre-defined keywords on batch inference
    I: stop frequency for each instance [int], keywords ids [list]
    '''
    def __init__(self, num, keywords_ids: list):
        self.num = num
        self.keywords = keywords_ids

    '''
    D: designed for early stopping with pre-defined keywords
    *N: currently we only support for one stop words, if you want to support multiple stop words, please modify the code
    on both keywords and frequency number
    '''
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for instance in input_ids:
            count = torch.sum(instance.eq(self.keywords[0])).item()
            if count >= self.num:
                pass
            else:
                return False
        return True

class PromptGenerator(object):
    def __init__(self):
        pass

    '''
    D: initial the model and tokenizer with the specific model type and device map
    I: model type [str], model name [str], device map [str](default is cuda), cache dir [str](default is '')
    O: model [AutoModelForCausalLM], tokenizer [AutoTokenizer]
    '''
    def initial_load(self, model_type, model_name, device_map='cuda', cache_dir=''):
        if model_type == 'peft':
            model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map=device_map,
                                                             torch_dtype=torch.bfloat16, cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
            tokenizer.pad_token = tokenizer.eos_token
        elif model_type == 'llm':
            device_map = {'': 0}
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=True,
                use_auth_token=True,
                cache_dir=cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError('model type should be peft or llm')
        return model, tokenizer

    '''
    D: generate the prompt with the specific model and tokenizer with early stop approval
    I: text [str], model [AutoModelForCausalLM], tokenizer [AutoTokenizer], stop words [list], device [str]
    O: output [str]
    *N: this is for case checking, for more efficient usage, please direct to the batch_generation
    '''
    def case_generate(self, text, model, tokenizer, stop_words, device, max_new_tokens=400):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        #inputs['input_ids'] = inputs['input_ids'][:, 1:]
        #inputs['attention_mask'] = inputs['attention_mask'][:, 1:]
        stop_ids = [tokenizer.encode(w)[-1] for w in stop_words]
        stop_criteria = KeywordsStoppingCriteriaCase(stop_ids)
        outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, stopping_criteria=StoppingCriteriaList([stop_criteria]))
        # for testing reason
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output

    '''
    D: generate the prompt with the specific model and tokenizer with early stop approval
    I: text [str], model [AutoModelForCausalLM], tokenizer [AutoTokenizer], stop words [list], device [str]
    O: output [str]
    *N: this is for case checking, for more efficient usage, please direct to the batch_generation
    '''
    def case_generate_iter(self, text, model, tokenizer, stop_words, device, max_new_tokens=400):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        # inputs['input_ids'] = inputs['input_ids'][:, 1:]
        # inputs['attention_mask'] = inputs['attention_mask'][:, 1:]
        stop_ids = [tokenizer.encode(w)[-1] for w in stop_words]
        stop_criteria = KeywordsStoppingCriteriaCase(stop_ids)
        outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"],
                                 max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                                 stopping_criteria=StoppingCriteriaList([stop_criteria]))
        # for testing reason
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if '�' in output:
            print('the error generation is:', output)
            output = self.case_generate_iter(text, model, tokenizer, stop_words, device, max_new_tokens+1)
        return output

    '''
    D: make the case generation controllable, support for two patterns:
    -- remove the first token patterns, which means the first token will be removed (the <s> token or token id general to be 1)
    -- add the tensor patterns, which means the tensor will be added before the input tensor (e.g. all padding token added to check the padding effect)
    I: text [str], model [AutoModelForCausalLM], tokenizer [AutoTokenizer], stop words [list], device [str]
    -- max_new_tokens [int], remove_f [bool], add_f [bool], add_tensor [tensor]
    O: output [str]
    '''
    def case_generate_control_input(self, text, model, tokenizer, stop_words, device, max_new_tokens=400, remove_f=False, add_f=False, add_tensor=None):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        if remove_f:
            inputs['input_ids'] = inputs['input_ids'][:, 1:]
            inputs['attention_mask'] = inputs['attention_mask'][:, 1:]
        if add_f:
            assert inputs['input_ids'].shape[0] == add_tensor.shape[0] # make the batch size equally
            inputs['input_ids'] = torch.cat((add_tensor.to(device), inputs['input_ids']), dim=0)
            inputs['attention_mask'] = torch.cat((torch.ones(add_tensor.shape).to(device), inputs['attention_mask']), dim=0)
        stop_ids = [tokenizer.encode(w)[-1] for w in stop_words]
        stop_criteria = KeywordsStoppingCriteriaCase(stop_ids)
        outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, stopping_criteria=StoppingCriteriaList([stop_criteria]))
        # for testing reason
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output

    '''
    D: replace the rest of the tokens after the last \n 13 with end token </s> 2
    I: the batch tensor [tensor], the specific number [int] (13)
    O: the replaced batch tensor [tensor]
    '''
    def replace_after_specific_batch(self, batch_tensor, num):
        # Loop over each tensor in the batch
        for i in range(batch_tensor.shape[0]):
            tensor = batch_tensor[i]

            # Reverse the tensor
            tensor_reversed = torch.flip(tensor, [0])
            # Find the index of the specific element in the reversed tensor
            index = (tensor_reversed == num).nonzero(as_tuple=True)[0][0]

            # Replace all elements after the specific element with 2
            tensor_reversed[:index] = 2

            # Reverse the tensor back to its original order
            tensor = torch.flip(tensor_reversed, [0])

            # Replace the tensor in the batch
            batch_tensor[i] = tensor

        return batch_tensor

    '''
    D: generate the prompt with the specific model and tokenizer with early stop approval with a batch support
    I: batch data[list(str)], model [AutoModelForCausalLM], tokenizer [AutoTokenizer], stop words [list], stop frequency for each instance [int] (e.g. 2 for two \n)
    --device [str(default is cuda)], batch size [int], max length [int]
    O: generation [list(str)]
    *N: this is a general batch generation, which means it can support any LLM model
    '''
    def batch_generation(self, batch_data, model, tokenizer, stop_words, stop_num, device, batch_size, max_length):
        generation = []
        tokenizer.pad_token = tokenizer.eos_token
        # mannual check for debugging
        # stop_words = ['\n']
        # stop_num = 2
        for start_index in tqdm(range(0, len(batch_data), batch_size)):
            end_index = min(start_index + batch_size, len(batch_data))
            encoded_batch = batch_data[start_index:end_index]
            inputs_batch = tokenizer(encoded_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            if stop_words and len(stop_words[0])>=1:
                stop_ids = [tokenizer.encode(w)[-1] for w in stop_words]
                #stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
                stop_criteria = KeywordsStoppingCriteriaBatch(stop_num, stop_ids)
                output_batch = model.generate(input_ids=inputs_batch["input_ids"].to(device), attention_mask=inputs_batch["attention_mask"].to(device), max_new_tokens=max_length,
                                              pad_token_id=tokenizer.eos_token_id,  stopping_criteria=StoppingCriteriaList([stop_criteria]))
            else:
                output_batch = model.generate(input_ids=inputs_batch["input_ids"].to(device), attention_mask=inputs_batch["attention_mask"].to(device), max_new_tokens=max_length,
                                              pad_token_id=tokenizer.eos_token_id)

            #output_batch = self.replace_after_specific_batch(output_batch, stop_ids[0])
            outs_batch = tokenizer.batch_decode(output_batch, skip_special_tokens=True)
            generation += outs_batch

            del inputs_batch
            del outs_batch
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        return generation

    '''
    D: generate the prompt with the specific model and tokenizer with early stop approval for prompt generation
    I: prompt data[list(str)], model [AutoModelForCausalLM], tokenizer [AutoTokenizer], stop words [list], stop frequency for each instance [int] (e.g. 2 for two \n)
    -- device [str(default is cuda)], batch size [int(default is 16)], max length [int(default is 400)]
    '''
    def template_generate(self, prompt_data, model, tokenizer, stop_words, stop_num, device='cuda', batch_size=16, max_length=400):
        prompt_generation = self.batch_generation(prompt_data, model, tokenizer, stop_words, stop_num, device, batch_size, max_length)
        original_prompt = prompt_data
        prompt_gene = dict()
        noprompt_gene = dict()
        for i, item in enumerate(prompt_generation):
            prompt_gene[i] = item
            noprompt_gene[i] = item.replace(original_prompt[i], '').strip()

        return prompt_gene, noprompt_gene


    '''
    D: generate the next token with the specific model and tokenizer, support for batch generation
    I: instance [list(str)], model [AutoModelForCausalLM], tokenizer [AutoTokenizer], mask index [int], device [str]
    -- expect [str](default is 'max', support for 'max' or 'all')
    O: next token id [tensor(int)], next token probability [tensor(float)] or all token probability [tensor(float)] 
    '''
    def next_token_only(self, instance, model, tokenizer, mask_idx, device, expect='max'):
        encoded_input = tokenizer(instance, return_tensors="pt").to(device)
        outputs = model(**encoded_input)
        probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
        if expect == 'max':
            # prob for the first tensor, argmax_prob for the second tensor
            argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
            return argmax_id, argmax_prob
        if expect == 'all':
            return probs



