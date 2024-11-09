import os, re, json
from tqdm import tqdm
import torch, numpy as np
import argparse
from baukit import TraceDict

# Include prompt creation helper functions
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.extract_utils import *
# this file from this git repo: https://github.com/ericwtodd/function_vectors
# if you directly use the code, please cite their work

def activation_replacement_per_class_intervention(prompt_data, avg_activations, dummy_labels, model, model_config, tokenizer, last_token_only=True):
    """
    Experiment to determine top intervention locations through avg activation replacement. 
    Performs a systematic sweep over attention heads (layer, head) to track their causal influence on probs of key tokens.

    Parameters: 
    prompt_data: dict containing ICL prompt examples, and template information
    avg_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    last_token_only: If True, only computes indirect effect for heads at the final token position. If False, computes indirect_effect for heads for all token classes

    Returns:   
    indirect_effect_storage: torch tensor containing the indirect_effect of each head for each token class.
    """
    device = model.device

    # Get sentence and token labels
    query_target_pair = prompt_data['query_target']

    query = query_target_pair['input']
    token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query=query)

    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
    idx_map = update_idx_map(idx_map, idx_avg)
      
    sentences = [prompt_string]# * model.config.n_head # batch things by head

    # Figure out tokens of interest
    tokens_of_interest = [query_target_pair['output']]
    if 'llama' in model_config['name_or_path']:
        ts = tokenizer(tokens_of_interest, return_tensors='pt').input_ids.squeeze()
        if tokenizer.decode(ts[1])=='' or ts[1]==29871: # avoid spacing issues
            token_id_of_interest = ts[2]
        else:
            token_id_of_interest = ts[1]
    else:
        token_id_of_interest = tokenizer(tokens_of_interest).input_ids[0][:1]
        
    inputs = tokenizer(sentences, return_tensors='pt').to(device)

    # Speed up computation by only computing causal effect at last token
    if last_token_only:
        token_classes = ['query_predictive']
        token_classes_regex = ['query_predictive_token']
    # Compute causal effect for all token classes (instead of just last token)
    else:
        token_classes = ['demonstration', 'label', 'separator', 'predictive', 'structural','end_of_example', 
                        'query_demonstration', 'query_structural', 'query_separator', 'query_predictive']
        token_classes_regex = ['demonstration_[\d]{1,}_token', 'demonstration_[\d]{1,}_label_token', 'separator_token', 'predictive_token', 'structural_token','end_of_example_token', 
                            'query_demonstration_token', 'query_structural_token', 'query_separator_token', 'query_predictive_token']
    

    indirect_effect_storage = torch.zeros(model_config['n_layers'], model_config['n_heads'],len(token_classes))

    # Clean Run of Baseline:
    clean_output = model(**inputs).logits[:,-1,:]
    clean_probs = torch.softmax(clean_output[0], dim=-1)

    # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
    for layer in range(model_config['n_layers']):
        head_hook_layer = [model_config['attn_hook_names'][layer]]
        
        for head_n in range(model_config['n_heads']):
            for i,(token_class, class_regex) in enumerate(zip(token_classes, token_classes_regex)):
                reg_class_match = re.compile(f"^{class_regex}$")
                class_token_inds = [x[0] for x in token_labels if reg_class_match.match(x[2])]

                intervention_locations = [(layer, head_n, token_n) for token_n in class_token_inds]
                intervention_fn = replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                           model=model, model_config=model_config,
                                                           batched_input=False, idx_map=idx_map, last_token_only=last_token_only)
                with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:                
                    output = model(**inputs).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
                
                # TRACK probs of tokens of interest
                intervention_probs = torch.softmax(output, dim=-1) # convert to probability distribution
                indirect_effect_storage[layer,head_n,i] = (intervention_probs-clean_probs).index_select(1, torch.LongTensor(token_id_of_interest).to(device).squeeze()).squeeze()

    return indirect_effect_storage


def compute_indirect_effect(dataset, mean_activations, model, model_config, tokenizer, n_shots=10, n_trials=25, last_token_only=True, prefixes=None, separators=None, filter_set=None):
    """
    Computes Indirect Effect of each head in the model

    Parameters:
    dataset: ICL dataset
    mean_activations:
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_shots: Number of shots in each in-context prompt
    n_trials: Number of in-context prompts to average over
    last_token_only: If True, only computes Indirect Effect for heads at the final token position. If False, computes Indirect Effect for heads for all token classes


    Returns:
    indirect_effect: torch tensor of the indirect effect for each attention head in the model, size n_trials * n_layers * n_heads
    """
    n_test_examples = 1

    if prefixes is not None and separators is not None:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, prefixes=prefixes, separators=separators)
    else:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer)

    is_llama = 'llama' in model_config['name_or_path']
    prepend_bos = not is_llama

    if last_token_only:
        indirect_effect = torch.zeros(n_trials,model_config['n_layers'], model_config['n_heads'])
    else:
        indirect_effect = torch.zeros(n_trials,model_config['n_layers'], model_config['n_heads'],10) # have 10 classes of tokens

    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))
    
    for i in tqdm(range(n_trials), total=n_trials):
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
        word_pairs_test = dataset['valid'][np.random.choice(filter_set,n_test_examples, replace=False)]
        if prefixes is not None and separators is not None:
            prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, shuffle_labels=True, 
                                                           prepend_bos_token=prepend_bos, prefixes=prefixes, separators=separators)
        else:
            prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, 
                                                           shuffle_labels=True, prepend_bos_token=prepend_bos)
        
        ind_effects = activation_replacement_per_class_intervention(prompt_data=prompt_data_random, 
                                                                    avg_activations = mean_activations, 
                                                                    dummy_labels=dummy_gt_labels, 
                                                                    model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                    last_token_only=last_token_only)
        indirect_effect[i] = ind_effects.squeeze()

    return indirect_effect


if __name__ == "__main__":
    # computation process for head average activation and indirect effect on heads
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded, delimited list input', type=str, required=True)
    # for multiple language pairs, use -dn "mt_de-en,mt_en-de,mt_en-zh,mt_zh-en"
    # or -dn mt_de-en,mt_en-de,mt_en-zh,mt_zh-en
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    # local name path or huggingface model name
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    # data root dir for the dataset, default is '../dataset_files'
    parser.add_argument('--save_path_root', help='File path to save indirect effect to', type=str, required=False, default='../results')
    # save path root for the  indirect effect
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type =int, required=False, default=10)
    parser.add_argument('--n_trials_ama', help="Number of in-context prompts to average over", type=int, required=False, default=1000)
    # the cases used for computing the average head activations

    parser.add_argument('--n_trials_cie', help="Number of in-context prompts for indirect effect to average over", type=int, required=False,
                        default=25)
    # the cases used for computing the indirect effect
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, type=float, default=0.3)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mean_activations_path', help='Path to mean activations file used for intervention', required=False, type=str, default=None)
    parser.add_argument('--last_token_only', help='Whether to compute indirect effect for heads at only the final token position, or for all token classes', required=False, type=bool, default=True)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    # you can customise this if you want to use your own template
    # e.g. --prefixes '{"input":"Q:", "output":"A:", "instructions":""}'
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})
    # you can customise this if you want to use your own template
    parser.add_argument('--format', help='Format of the prompt template', type=str, required=False, default='qa')
    # default is the qa format from FUNCTION VECTOR, we use the lang format as well for a robust testing
    args = parser.parse_args()
    '''
    exp command
    --dataset_name mt_de-en,mt_en-de,mt_en-zh,mt_zh-en \
    --model_name /mnt/sgnfsdata/tolo-02-95/weicwang/model/llama2-7b-hf \
    --root_data_dir ../dataset_files \
    --save_path_root ../cache \
    --seed 42 \
    --n_shots 10 \
    --n_trials_ama 1000 \
    --n_trials_cie 25 \
    --test_split 0.3 \
    --device cuda \
    --last_token_only True \
    --mean_activations_path ../cache \
    --format qa \
    #--prefixes {"input":"Q:", "output":"A:", "instructions":""} \
    #--separators {"input":"\n", "output":"\n", "instructions":""} \
    '''
    # computation list
    dataset_names = [str(item) for item in args.dataset_name.split(',')]
    #dataset_name = 'english-german'
    model_name = args.model_name
    root_data_dir = args.root_data_dir

    seed = args.seed
    n_shots = args.n_shots
    n_trials_ama = args.n_trials_ama
    n_trials_cie = args.n_trials_cie
    test_split = args.test_split
    device = args.device
    lang2language = {'en': 'English', 'de': 'German', 'zh': 'Chinese'}

    save_path_root = args.save_path_root
    mean_activations_path = args.mean_activations_path

    last_token_only = args.last_token_only
    if args.format in ['qa', 'lang']:
        if args.format == 'qa':
            prefixes = {"input":"Q:", "output":"A:", "instructions":""}
            separators = {"input":"\n", "output":"\n", "instructions":""}
        else:

            prefixes = {"input":"{src_lang}:", "output":"{trg_lang}:", "instructions":""}
            separators = {"input":"\n", "output":"\n", "instructions":""}
    else:
        prefixes = args.prefixes
        separators = args.separators



    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    set_seed(seed)
    mean_activations_path = args.mean_activations_path
    for dataset_name in dataset_names:


        dataset_name = dataset_name
        # Load the dataset
        print("Loading Dataset")
        dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

        language_save_pairs = dataset_name.split('.')[0]
        # specific the save path root
        save_path_root_specific = f"{save_path_root}/{language_save_pairs}"
        src_lang = dataset_name.split('_')[-1].split('-')[0]
        trg_lang = dataset_name.split('_')[-1].split('-')[1]

        if args.format=='lang':
            src_lang = lang2language[src_lang]
            trg_lang = lang2language[trg_lang]


        if not os.path.exists(save_path_root_specific):
            os.makedirs(save_path_root_specific)

        # Load or Re-Compute Mean Activations
        if os.path.exists(f'{mean_activations_path}/{dataset_name}_mean_head_activations_{args.format}.pt'):
            print(f"Loading Mean Activations from {mean_activations_path}/{dataset_name}_mean_head_activations_{args.format}.pt")
            mean_activations_path_lang = f'{mean_activations_path}/{dataset_name}_mean_head_activations_{args.format}.pt'
            mean_activations = torch.load(mean_activations_path_lang)
        else:
            print(f"Computing Mean Activations for {dataset_name}")
            mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer,
                                                         n_icl_examples=n_shots, N_TRIALS=n_trials_ama, prefixes=prefixes, separators=separators)
            torch.save(mean_activations, f'{mean_activations_path}/{dataset_name}_mean_head_activations_{args.format}.pt')


        print(f"Computing Indirect Effect for {dataset_name}")
        indirect_effect = compute_indirect_effect(dataset, mean_activations, model=model, model_config=model_config, tokenizer=tokenizer,
                                                  n_shots=n_shots, n_trials=n_trials_cie, last_token_only=last_token_only, prefixes=prefixes, separators=separators)


        torch.save(indirect_effect, f'{save_path_root_specific}/{dataset_name}_indirect_effect_{args.format}.pt')

    