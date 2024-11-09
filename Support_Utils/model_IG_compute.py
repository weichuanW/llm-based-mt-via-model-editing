import torch
import einops
from .model_inner_intervention import replace_batch_activations
from .model_inner_extraction import gather_attn_activations_batch
from baukit import TraceDict
import torch.nn.functional as F
import numpy as np
'''
D: scaling the input alone the batch dimension, gradually scaling them over `steps` steps from 0 to their original value over the batch dimensions.
I: activations [torch.Tensor] (n_token, dim), steps [int], device [str]
O: out [torch.Tensor] (steps, n_token, dim)
'''
def scaled_input(activations: torch.Tensor, steps=20, device='cpu'):
    tiled_activations = einops.repeat(activations, "b d -> r b d", r=steps) # extend from (layer, dim) to (step, layer, dim)
    #print(tiled_activations.shape)
    out = (
            tiled_activations
            * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None, None]
    )
    return out

'''
D: compute the internal integral of the model with a specific activation module (the input for a transformation matrix)
I: prompt [str/dict] the dict should contain input_ids, and attention_mask with tensor format
-- exp_token_id (generated for the prompy) [int], model [AutoModelForCausalLM], tokenizer [AutoTokenizer], model_config [dict], 
-- layers [int], replace_module [str] (for any intermediate hidden states)
-- batch_size [int], steps [int], the batch size used for real running to avoid the memory issues, the real chunk is computed by steps // batch_size
-- attribution_method [str] (default is "integrated_grads", further can design to support more integration methods)
*N: case-level IG computation
'''
def internal_integral(prompt, exp_token_id,  model, tokenizer, model_config, layers=32, replace_module=None, batch_size=5, steps=20):
    assert replace_module !=None
    # get the intermediate representation of the model with the TraceDict
    if isinstance(prompt, str):
        # todo: if the gpt memory not enough, please mapping this part to the CPU
        activation_td = gather_attn_activations_batch([prompt], model_config[replace_module], model, tokenizer)
        # initial the prompt
        encoded_input = tokenizer(prompt, return_tensors='pt').to(model.device)
    elif isinstance(prompt, dict):
        prompt['input_ids'] = prompt['input_ids'].to(model.device)
        prompt['attention_mask'] = prompt['attention_mask'].to(model.device)

        activation_td = gather_attn_activations_batch(prompt, model_config[replace_module], model, tokenizer)
        # initial the prompt
        encoded_input = prompt
    else:
        raise ValueError('prompt should be either string or dict')
    # original shape should be (1, n_token, dim) 1 for the batch size
    # after vstack, the size should be (n_layer, n_token, dim)
    # layer for specific layer module name
    activation_inner_repre = torch.vstack([activation_td[layer].input for layer in model_config[replace_module]]) # baseline activations for all layers
    activation_inner_last = activation_inner_repre[:, -1, :] # size to be (n_layer, dim)

    # scaling the intermediate representation with batch setting
    scaled_weights = scaled_input(activation_inner_last, steps=steps, device=model.device) # size to be (steps, n_layer, dim), the steps can be regarded as batch
    scaled_weights.requires_grad_(True)

    chunk_num = steps // batch_size
    # initial the prompt
    # encoded_input = tokenizer(prompt, return_tensors='pt').to(model.device)

    # for gradient concatenation
    integrated_grads_this_step = []
    for batch_weights in scaled_weights.chunk(chunk_num):
        # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
        # first tile the inputs to the correct batch size
        # support for CLM models / decoder only models
        #print('encoded_input shape is {}'.format(encoded_input["input_ids"].shape))
        #print('attention_mask shape is {}'.format(encoded_input["attention_mask"].shape))
        inputs = {
            "input_ids": einops.repeat(
                encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
            ),
            "attention_mask": einops.repeat(
                encoded_input["attention_mask"],
                "b d -> (r b) d",
                r=batch_size,
            ),
        }

        # replace the activations with the setting activations
        # make the gradient to be retained
        scaled_fn = replace_batch_activations(layers, batch_weights, model, model_config,
                                              replace_module=replace_module, token_id=-1)
        with TraceDict(model, layers=model_config[replace_module], edit_output=scaled_fn, retain_grad=True):
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits[:, -1, :], dim=-1) # last token only (batch_size, vocab_size)

            grad = torch.autograd.grad(torch.unbind(probs[:, exp_token_id]), batch_weights)[0]
            grad_ = grad.sum(dim=0).detach().cpu()
            integrated_grads_this_step.append(grad_)
            del grad, batch_weights
            torch.cuda.empty_cache()
    # summation
    integrated_grads_this_step = torch.stack(integrated_grads_this_step, dim=0).sum(dim=0) # summation the batch gradient, (n_layer, act_dim)
    baseline_activations = activation_inner_last.detach().cpu() # (layers, act_dim)

    # division
    integrated_grads_this_step *= baseline_activations.squeeze(0) / steps # element-wise multiplication (layers, act_dim)

    return integrated_grads_this_step

'''
D: return the top k elements and corresponding 2-D indices of a tensor
'''
def top_k_elements(tensor_, k):
    # Convert the tensor to a numpy array
    array = tensor_.detach().cpu().numpy()

    # Flatten the array and get the indices of the top k values
    flat_indices = np.argpartition(array.ravel(), -k)[-k:]
    flat_indices_sorted = flat_indices[np.argsort(-array.ravel()[flat_indices])]

    # Convert the flat indices to 2D indices
    indices = np.unravel_index(flat_indices_sorted, array.shape)

    # Get the top k values
    values = array[indices]
    tuple_indices = tuple(map(tuple, indices))

    # organise the tuple indices
    tuple_indices = [[tuple_indices[0][i], tuple_indices[1][i]] for i in range(len(tuple_indices[0]))]
    return values, tuple_indices
