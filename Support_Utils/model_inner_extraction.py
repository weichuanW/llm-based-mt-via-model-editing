import torch
from baukit import TraceDict
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from tqdm import tqdm
import numpy as np

# Attention Activations
'''
D: get all attention activations for one prompt sentence based on the baukit TraceDict method with the batch setting
-- for the input part, which is usually called activations (before the matrix projection)
I: prompt sentence [list(str)], layers name from config [list], model object [huggingface model],
-- tokenizer object [huggingface tokenizer]
-- max length for the tokenization [int] (default is 1024)
O: mean token activation for a specific set [n_layer, n_head, head_dim] with the numpy format
*N: the TraceDict can get a batch result with the padding operation, please notice this
-- however, the tracing does not make inference, so the computation is quite low
-- please notice this provide the input part for a specific module
'''
def gather_attn_activations_batch(prompt_instances, layers, model, tokenizer, max_length=1024):
    if isinstance(prompt_instances, str):
        inputs = tokenizer(prompt_instances, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
    elif isinstance(prompt_instances, list):
        inputs = tokenizer(prompt_instances, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
    elif isinstance(prompt_instances, dict):
        inputs = prompt_instances
    else:
        raise ValueError('Please provide a valid prompt instance')
    # Access Activations
    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:
        model(**inputs) # batch_size x n_tokens x residual dimension for inner representation input, only want the inner representation

    return td

'''
D: calculate the token account with the token indices
I: a batch of token indices [list(int/tuple/list)]
O: selection token number by the token indices [int]
'''
def batch_token_account(token_indices):
    token_count = 0
    if isinstance(token_indices[0], int):
        for item in token_indices:
            assert item < 0
            token_count += abs(item)
    # tuple for a span calculation [item[0]: item[1]]
    elif isinstance(token_indices[0], tuple):
        for item in token_indices:
            token_count += abs(item[0] - item[1])
    # list for specific calculation [item[0], item[1], ...]
    elif isinstance(token_indices[0], list):
        for item in token_indices:
            token_count += len(item)
    else:
        raise ValueError('Please provide a valid token indices list')
    assert token_count != 0

    return token_count

'''
D: handling the token indices selection from a inner representation with batch setting
I: batched indices: [list(int/tuple/list)], 
-- activated inner representation for a specific layer [tensor]
-- [batch, token, ...], reshape for the final output [tuple] (default is (-1, ...))
O: the selected numpy data 
*N: this is for the layer calculation, which measn the each input activation is for a specific layer
-- so please not include the layer parameter in the re_shape
'''
def batch_token_indices_selection(encoded_indices, activation_current_layer, re_shape):
    if isinstance(encoded_indices[0], int):
        # reshape it to the (batch * token_cases, n_heads, head_dim)
        layer_tokens_repres = torch.vstack([activation_current_layer[batch_now][item:] for (batch_now, item) in
                                            enumerate(encoded_indices)]).view(re_shape).detach().cpu().numpy()

    # tuple for a span calculation [item[0]: item[1]]
    elif isinstance(encoded_indices[0], tuple):
        layer_tokens_repres = torch.vstack(
            [activation_current_layer[batch_now][item[0]: item[1]] for (batch_now, item) in
             enumerate(encoded_indices)]).view(re_shape).detach().cpu().numpy()
    # list for specific calculation [item[0], item[1], ...]

    elif isinstance(encoded_indices[0], list):
        # torch.Tensor(item).to(model.device)
        layer_tokens_repres = torch.vstack([activation_current_layer[batch_now][item] for (batch_now, item) in
                                            enumerate(encoded_indices)]).view(re_shape).detach().cpu().numpy()
    else:
        raise ValueError('No acceptable data type for indices')
    return layer_tokens_repres


'''
D: get the mean MLP activation kn representation part with a dataset object (batch version)
I: dataset object [dataset], huggingface model object, model_config [dict], huggingface tokenizer
-- token indices (list[int], [list(tuple)] or [list(list)]), if tuple, which lead to the beginning and end indices, 
-- int [item: ], tuple [item[0]: item[1]], list [item[0], item[1], ...]
-- batch size for computation [int], default is 1
-- kn_hook_name, the setting for hooked names for knowledge neurons
O: mean activations for KN dimension with the size [layer, kn_dimension] (mean for multiple instances)
'''
def get_mean_mlp_activations_kn_batch(dataset, model, model_config, tokenizer, token_indices, batch_size=1, kn_hook_name=None):
    if not kn_hook_name:
        raise ValueError('Please provide the knowledge neuron hook names')
    token_count = batch_token_account(token_indices)

    assert 'n_layers' in model_config
    assert 'kn_dim' in model_config
    # activation storage with numpy for size (overall token number, n_layers, kn_dim)
    activation_storage = np.zeros((token_count, model_config['n_layers'], model_config['kn_dim']))

    overall_token_begin = 0
    overall_token_end = 0
    for start_index in tqdm(range(0, len(dataset['train']), batch_size), desc='batch running on mean kn extraction'):
        end_index = min(start_index + batch_size, len(dataset['train']))
        encoded_batch = dataset['train']['input'][start_index:end_index]

        if len(token_indices) ==1:
            encoded_indices = token_indices
        else:
            encoded_indices = token_indices[start_index:end_index]
        activations_td = gather_attn_activations_batch(prompt_instances=encoded_batch,
                                                       layers=model_config[kn_hook_name],
                                                       model=model,
                                                       tokenizer=tokenizer)

        for i, layer in enumerate(model_config[kn_hook_name]):
            activation_current_layer = activations_td[layer].input
            layer_id = i

            # handling indices and transfer them into numpy
            layer_tokens_repres = batch_token_indices_selection(encoded_indices, activation_current_layer, (
            -1, model_config['kn_dim']))

            if layer_id == 0:
                # update the token end with the layer 0 beginning
                overall_token_end += layer_tokens_repres.shape[0]
            activation_storage[overall_token_begin: overall_token_end, layer_id] = layer_tokens_repres

            if layer_id == len(model_config[kn_hook_name]) - 1:
                # update the token begin with the layer all finish for next
                overall_token_begin += layer_tokens_repres.shape[0]

    # means on all instances
    mean_activations = activation_storage.mean(axis=0)

    # please add activation_storage for the output if you want to do further analysis
    return mean_activations



'''
D: get the mean head activations with a dataset object (batch version)
I: dataset object [dataset], huggingface model object, model_config [dict], huggingface tokenizer
-- token indices (list[int], [list(tuple)] or [list(list)]), if tuple, which lead to the beginning and end indices, 
-- int [item: ], tuple [item[0]: item[1]], list [item[0], item[1], ...]
-- batch size for computation [int], default is 1
O: mean activations with the size [layer, head, head_dimension] (mean for multiple instances)
*N: for a random selection, dataset['train'][np.random.choice(len(dataset['train']), N_TRIALS, replace=False)]
-- for tuple indices, we directly use the [begin: end] to extract, please notice that the batch tokenizer is left padding,
-- which means you would be better to consider the indices from right to left
-- e.g. [-10: -2] for (-10, -2), [-10: ] for -10, 
-- if you want to use a subset of the dataset, please pre-handling before calling this function
--e.g. prompt_random_instances = dataset['train'][np.random.choice(len(dataset['train']), N_TRIALS, replace=False)]
'''
def get_mean_head_activations_batch(dataset, model, model_config, tokenizer, token_indices, batch_size=1):
    assert 'n_heads' in model_config
    assert 'resid_dim' in model_config
    assert 'n_layers' in model_config

    '''
    D: reshape the activation shape to fulfill the shape
    '''
    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations

    # initial the storage tables
    # using np to storage the states

    # calculate the token number
    # int for a span calculation [item:]
    token_count = batch_token_account(token_indices)

    # activation storage with numpy for size (overall token number, n_layers, n_heads, head_dim)
    activation_storage = np.zeros((token_count, model_config['n_layers'], model_config['n_heads'], model_config['resid_dim'] // model_config['n_heads']))

    overall_token_begin = 0
    overall_token_end = 0
    for start_index in tqdm(range(0, len(dataset['train']), batch_size), desc='batching running on mean head extraction'):
        end_index = min(start_index + batch_size, len(dataset['train']))
        encoded_batch = dataset['train']['input'][start_index:end_index]
        if len(token_indices) ==1:
            encoded_indices = token_indices
        else:
            encoded_indices = token_indices[start_index:end_index]


        activations_td = gather_attn_activations_batch(prompt_instances=encoded_batch,
                                                       layers=model_config['attn_hook_names'],
                                                       model=model,
                                                       tokenizer=tokenizer)

        for i, layer in enumerate(model_config['attn_hook_names']):
            activation_current_layer = split_activations_by_head(activations_td[layer].input, model_config)
            layer_id = i

            # handling indices and transfer them into numpy
            layer_tokens_repres = batch_token_indices_selection(encoded_indices, activation_current_layer, (-1, model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']))

            if layer_id == 0:
                # update the token end with the layer 0 beginning
                overall_token_end += layer_tokens_repres.shape[0]
            activation_storage[overall_token_begin: overall_token_end, layer_id] = layer_tokens_repres

            if layer_id == len(model_config['attn_hook_names'])-1:
                # update the token begin with the layer all finish for next
                overall_token_begin += layer_tokens_repres.shape[0]

    # means on all instances
    mean_activations = activation_storage.mean(axis=0)
    print(111)
    # please add activation_storage for the output if you want to do further analysis
    return mean_activations

# Layer Activations
'''
D: get all layer activations for one prompt sentence based on the baukit TraceDict method with the batch setting
-- get the output of a specific layer, which means the layer output
I: prompt sentence [str], layers name from config [list], model object [huggingface model],
-- tokenizer object [huggingface tokenizer]
-- max length for the tokenization [int] (default is 1024)
O: tracedict with stored activations
*N: please notice that this provide the output part for a specific module
'''
def gather_layer_activations_batch(prompt_instances, layers, model, tokenizer, max_length=1024):

    inputs = tokenizer(prompt_instances, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(
        model.device)

    # Access Activations
    with TraceDict(model, layers=layers, retain_input=False, retain_output=True) as td:
        model(**inputs) # batch_size x n_tokens x residual dimension for inner representation output, only want the inner representation

    return td

'''
D: get the mean layer activations with a dataset object for specific tokens with the batch setting
I: dataset object [dataset], huggingface model object [model], model_config [dict], huggingface tokenizer [tokenizer]
-- token indices (list[int], [list(tuple)] or [list(list)]), if tuple, which lead to the beginning and end indices, 
-- int [item: ], tuple [item[0]: item[1]], list [item[0], item[1], ...]
-- batch size for computation [int], default is 1
-- layer name: for the specific output of the layer name 
-- e.g. f'model.layers.{layer}.self_attn.o_proj', f'model.layers.{layer}.mlp.down_proj' rather than
-- f'model.layers.{layer}'
O: mean activations layer output with the size [layer, residual dimension] (mean for multiple instances)
'''
def get_mean_layer_activations_batch(dataset, model, model_config, tokenizer, token_indices, batch_size=1, layer_name=None):
    if layer_name==None:
        raise ValueError('Please specific a layer name for inner calculation')

    token_count = batch_token_account(token_indices)

    assert 'resid_dim' in model_config
    assert 'n_layers' in model_config

    # activation storage with numpy for size (overall token number, n_layers, n_heads, head_dim)
    activation_storage = np.zeros((token_count, model_config['n_layers'], model_config['resid_dim']))

    overall_token_begin = 0
    overall_token_end = 0
    for start_index in tqdm(range(0, len(dataset['train']), batch_size), desc='batching running on mean layer extraction'):
        end_index = min(start_index + batch_size, len(dataset['train']))
        encoded_batch = dataset['train']['input'][start_index:end_index]
        if len(token_indices) ==1:
            encoded_indices = token_indices
        else:
            encoded_indices = token_indices[start_index:end_index]

        activations_td = gather_layer_activations_batch(prompt_instances=encoded_batch,
                                                  layers=model_config[layer_name],
                                                  model=model,
                                                  tokenizer=tokenizer)
        for i, layer in enumerate(model_config[layer_name]):
            # shape is (batch, token_num, dimension)
            activation_current_layer = activations_td[layer].output[0]
            layer_id = i

            layer_tokens_repres = batch_token_indices_selection(encoded_indices, activation_current_layer, (-1, model_config['resid_dim']))

            if layer_id == 0:
                # update the token end with the layer 0 beginning
                overall_token_end += layer_tokens_repres.shape[0]
            print(activation_current_layer.shape)
            print(layer_tokens_repres.shape)
            activation_storage[overall_token_begin: overall_token_end, layer_id] = layer_tokens_repres


            if layer_id == len(model_config[layer_name])-1:
                # update the token begin with the layer all finish for next
                overall_token_begin += layer_tokens_repres.shape[0]
    # means on all instances
    mean_activations = activation_storage.mean(axis=0)

    # please add activation_storage for the output if you want to do further analysis
    return mean_activations

# Attention Weights
'''
D: get the multi-head attention weights and the corresponding all weighted values
-- using p=2 norm
-- this is used to draw the task attention visualisation heatmap
I: sentence [str], model object [huggingface model], model_config [dict], tokenizer object [huggingface tokenizer]
O: attention weights [torch.tensor], value weighted attention weights [torch.tensor]
'''
def get_value_weighted_attention(sentence, model, model_config, tokenizer):
    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
    output = model(**inputs, output_attentions=True) # batch_size x n_tokens x vocab_size, only want last token prediction
    attentions = torch.vstack(output.attentions) # (layers, heads, tokens, tokens)
    values = torch.vstack([output.past_key_values[i][1] for i in range(model_config['n_layers'])]) # (layers, heads, tokens, head_dim)
    value_weighted_attn = torch.einsum("abcd,abd->abcd", attentions, values.norm(dim=-1))
    return attentions, value_weighted_attn
