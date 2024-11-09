from baukit import TraceDict, get_module
import torch
import re
import bitsandbytes as bnb

def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


'''
D: replace the attention activation with the average vectors
-- modify version, only for the last token modification
-- used for attention design, which means we have another activation for the MLP which is called Knowledge Neurons (KNs)
I: layer_head_token_pairs [list[tuple]], avg_activations [torch.Tensor], model [AutoModelForCausalLM], model_config [dict], act_replace_id [int],
-- batched_input [bool], last_token_only [bool]
-- operations [str] (default is None) valid set (add, subtract, replace, multiply)
-- device [str] (default is 'cuda')
O: rep_act [function]
'''
def inter_activation_attn_head(layer_head_token_pairs, avg_activations, model, model_config, act_replace_id=-1,
                                last_token_only=False, operations=None, device='cuda'):
    edit_layers = [x[0] for x in layer_head_token_pairs]

    # add support for batch inference / modification on the batch inputs [str1, str2, str3, ...]
    def inter_act(output, layer_name, inputs):
        if operations is not None:
            current_layer = int(layer_name.split('.')[2])
            if current_layer in edit_layers:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]

                # Determine shapes for intervention
                original_shape = inputs.shape
                new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim'] // model_config[
                    'n_heads'])  # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
                inputs = inputs.view(*new_shape)  # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

                batch_size = original_shape[0]
                for i in range(batch_size):
                    if last_token_only:
                        # Patch activations only at the last token for interventions like
                        for (layer, head_n, token_n) in layer_head_token_pairs:
                            if layer == current_layer:
                                if operations == 'add':
                                    #print('intervention')
                                    inputs[i, -1, head_n] += avg_activations[layer, head_n, act_replace_id].to(device)
                                elif operations == 'subtract':
                                    inputs[i, -1, head_n] -= avg_activations[layer, head_n, act_replace_id].to(device)
                                elif operations == 'replace':
                                    inputs[i, -1, head_n] = avg_activations[layer, head_n, act_replace_id].to(device)
                                elif operations == 'multiply':
                                    inputs[i, -1, head_n] *= avg_activations[layer, head_n, act_replace_id].to(device)
                                elif operations == 'suppress':
                                    inputs[i, -1, head_n] = inputs[i, -1, head_n].zero_()
                                elif operations == 'enhance':
                                    inputs[i, -1, head_n] = inputs[i, -1, head_n] * 2
                                else:
                                    raise ValueError("Invalid operation")
                    else:
                        # Patch activations into baseline sentence found at index, -1 of the batch (targeted & multi-token patching)
                        for (layer, head_n, token_n) in layer_head_token_pairs:
                            if layer == current_layer:
                                if operations == 'add':
                                    inputs[i, token_n, head_n] += avg_activations[layer, head_n, act_replace_id].to(device)
                                elif operations == 'subtract':
                                    inputs[i, token_n, head_n] -= avg_activations[layer, head_n, act_replace_id].to(device)
                                elif operations == 'replace':
                                    inputs[i, token_n, head_n] = avg_activations[layer, head_n, act_replace_id].to(device)
                                elif operations == 'multiply':
                                    inputs[i, token_n, head_n] *= avg_activations[layer, head_n, act_replace_id].to(device)
                                elif operations == 'suppress':
                                    inputs[i, token_n, head_n] = inputs[i, token_n, head_n].zero_()
                                elif operations == 'enhance':
                                    inputs[i, token_n, head_n] = inputs[i, token_n, head_n] * 2
                                else:
                                    raise ValueError("Invalid operation")
                inputs = inputs.view(*original_shape)
                proj_module = get_module(model, layer_name)
                out_proj = proj_module.weight

                # since we get the activation representation, so we need to further multiple the attention Whole projection
                if 'llama' in model_config['name_or_path']:
                    if '70b' in model_config['name_or_path']:
                        # need to dequantize weights
                        out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
                        new_output = torch.matmul(inputs, out_proj_dequant.T)
                    else:
                        new_output = torch.matmul(inputs, out_proj.T)

                return new_output
        else:
            return output

    return inter_act

'''
D: Adds a vector to the output of a specified layer in the model
I: edit_layer [int], fv_vector [torch.Tensor], device [str], 
-- operation [str] (default is None) valid set (add, subtract, replace, multiply)
-- idx [int] (default is -1, the last token)
O: add_act [function]
'''
def inter_output_vector(edit_layer, fv_vector, device, operation=None, idx=-1):
    def inter_act(output, layer_name):
        if operation is not None:
            current_layer = int(layer_name.split(".")[2])
            if current_layer == edit_layer:
                if isinstance(output, tuple):
                    if operation == "add":
                        output[0][:, idx] += fv_vector.to(device)
                    elif operation == "subtract":
                        output[0][:, idx] -= fv_vector.to(device)
                    elif operation == "replace":
                        output[0][:, idx] = fv_vector.to(device)
                    elif operation == "multiply":
                        output[0][:, idx] *= fv_vector.to(device)
                    else:
                        raise ValueError("Invalid operation")
                    return output
                elif isinstance(output, torch.Tensor):
                    if operation == "add":
                        output[:, idx] += fv_vector.to(device)
                    elif operation == "subtract":
                        output[:, idx] -= fv_vector.to(device)
                    elif operation == "replace":
                        output[:, idx] = fv_vector.to(device)
                    elif operation == "multiply":
                        output[:, idx] *= fv_vector.to(device)
                    else:
                        raise ValueError("Invalid operation")
                    return output
                else:
                    return output
        else:
            return output

    return inter_act

'''
D: editing the KN activation with modified KN position
I: edit_dict [dict[int: list]], model config, 
-- mode [str] (suppress or enhance)
-- device [str], idx [int]
'''
def edit_kn_samemode(edit_dict, model, model_config, device, mode=None, idx=-1):
    def edit_act(output, layer_name, inputs):
        if layer_name in [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.config.num_hidden_layers)]:
            current_layer = int(layer_name.split(".")[2])
            if current_layer in edit_dict:
                assert mode!=None
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                if isinstance(inputs, torch.Tensor):
                    # modified the inputs
                    ori_inputs = inputs[:, idx, edit_dict[current_layer]].detach().clone()
                    if mode == 'suppress':
                        ori_output = ori_inputs.zero_()
                        inputs[:, idx, edit_dict[current_layer]] = ori_output.to(device)
                    elif mode == 'enhance':
                        # extend value to double, which means add them again
                        #print('success editing')
                        inputs[:, idx, edit_dict[current_layer]] += ori_inputs.to(device)
                    else:
                        raise ValueError('mode should be suppress or enhance')
                    # get the weight of the down_proj
                    proj_module = get_module(model, layer_name)
                    out_proj = proj_module.weight

                    # computation
                    if 'gpt2-xl' in model_config[
                        'name_or_path']:  # GPT2-XL uses Conv1D (not nn.Linear) & has a bias term, GPTJ does not
                        out_proj_bias = proj_module.bias
                        new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)

                    elif 'gpt-j' in model_config['name_or_path']:
                        new_output = torch.matmul(inputs, out_proj.T)

                    elif 'gpt-neox' in model_config['name_or_path']:
                        out_proj_bias = proj_module.bias
                        new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)

                    elif 'llama' in model_config['name_or_path']:
                        if '70b' in model_config['name_or_path']:
                            # need to dequantize weights
                            out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
                            new_output = torch.matmul(inputs, out_proj_dequant.T)
                        else:
                            #print('finishing on this part for final computation')
                            new_output = torch.matmul(inputs, out_proj.T)

                    return new_output
        else:
            return output

    return edit_act

'''
D: editing the KN activation with modified KN position
I: edit_dict [dict[int: list[tuple]]], model config, 
-- mode [str] (suppress or enhance)
-- device [str], idx [int]
E: for a edit_dict, we have {0:[('suppress', 1), ('enhance', 2), ('suppress', 3)]}
'''
def edit_kn(edit_dict, model, model_config, idx=-1):
    """
    modified the KN activation with the modified KN position

    Returns:
    add_act: a function specifying how to add a function vector to a layer's output hidden state
    """
    def add_act(output, layer_name, inputs):
        if layer_name in [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.config.num_hidden_layers)]:
            current_layer = int(layer_name.split(".")[2])
            if current_layer in edit_dict:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                if isinstance(inputs, torch.Tensor):
                    # modified the inputs
                    kns = edit_dict[current_layer]
                    # do a kn-level modification
                    for kn_item in kns:
                        mode, kn_idx = kn_item

                    # initial kNs positions
                        if mode == 'suppress':
                            inputs[:, idx, kn_idx] = inputs[:, idx, kn_idx].zero_()
                        elif mode == 'enhance':
                            # extend value to double, which means add them again
                            inputs[:, idx, kn_idx] = inputs[:, idx, kn_idx] * 2
                        else:
                            raise ValueError('mode should be suppress or enhance')
                        # get the weight of the down_proj
                    proj_module = get_module(model, layer_name)
                    out_proj = proj_module.weight

                    # computation
                    if 'gpt2-xl' in model_config[
                        'name_or_path']:  # GPT2-XL uses Conv1D (not nn.Linear) & has a bias term, GPTJ does not
                        out_proj_bias = proj_module.bias
                        new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)

                    elif 'gpt-j' in model_config['name_or_path']:
                        new_output = torch.matmul(inputs, out_proj.T)

                    elif 'gpt-neox' in model_config['name_or_path']:
                        out_proj_bias = proj_module.bias
                        new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)

                    elif 'llama' in model_config['name_or_path']:
                        if '70b' in model_config['name_or_path']:
                            # need to dequantize weights
                            out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
                            new_output = torch.matmul(inputs, out_proj_dequant.T)
                        else:
                            #print('finishing on this part for final computation')
                            new_output = torch.matmul(inputs, out_proj.T)

                    return new_output
        else:
            return output

    return add_act

'''
D: replace the activation with the corresponding layers
-- please note that the activations generally will be the input for specific layer / module, while in the frame of 
-- baukit, the hook function needs to return to the output (for personal understanding, it just like the forward function)
-- so if you just input the input part, it does not finish the forward action for the hook function, please also read
-- or call the output transfer matrix to finish the output part
I: layers for replacement [int] (default for all layers) / [list(int)] a list of modified layers
-- replace_batch_activations [torch.Tensor] (batch, layers, dim), 
-- model [huggingface model], model_config [dict], replace_module [str], token_id [int] (default is -1)
*N: current we only support the last token replacement, for other token replacement, please modify the code about the token_id
'''
def replace_batch_activations(layers, replace_batch_activations, model, model_config, replace_module=None, token_id=-1):
    if isinstance(layers, int):
        layers_ids = [layer_ for layer_ in range(layers)]
    if isinstance(layers, list):
        layers_ids = layers
    layer_modules = model_config[replace_module]
    def add_act(output, layer_name, inputs):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in layers_ids:
            if layer_name in layer_modules:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                # inputs (batch, n_token, residual_dim)
                # replacement with the current layer activations
                inputs[:, token_id] = replace_batch_activations[:, current_layer].to(model.device)

                # get the final transfermation matrix
                proj_module = get_module(model, layer_name)
                out_proj = proj_module.weight

                # since we get the activation representation, so we need to further multiple the attention Whole projection
                if 'gpt2-xl' in model_config[
                    'name_or_path']:  # GPT2-XL uses Conv1D (not nn.Linear) & has a bias term, GPTJ does not
                    out_proj_bias = proj_module.bias
                    new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)

                elif 'gpt-j' in model_config['name_or_path']:
                    new_output = torch.matmul(inputs, out_proj.T)

                elif 'gpt-neox' in model_config['name_or_path']:
                    out_proj_bias = proj_module.bias
                    new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)

                elif 'llama' in model_config['name_or_path']:
                    if '70b' in model_config['name_or_path']:
                        # need to dequantize weights
                        out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
                        new_output = torch.matmul(inputs, out_proj_dequant.T)
                    else:
                        new_output = torch.matmul(inputs, out_proj.T)

                return new_output
            else:
                return output

        else:
            return output
    return add_act