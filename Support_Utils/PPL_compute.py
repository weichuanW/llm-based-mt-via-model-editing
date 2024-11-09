def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    # print('input_encs:', token_lists)
    maxlen = max(len(t) for t in token_lists)
    # padding_side='left'
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # print('length of input_ids:', maxlen)
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

p(token)
p(span)

def get_ppl(model, tokenizer, prefix, target):
    '''
    e.g., prefix = "I find this movie very interesting" # normal part
    target = "very interesting" # repetition part
    '''

    '''
    I: i have a apple *20
    KN kn*0, k*0.05
    
    '''
    target_temp = target
    prompt = prefix + target_temp
    prompt_tokens = tokenizer.encode(prompt)
    prefix_tokens = tokenizer.encode(prefix)
    output_tokens = tokenizer.encode(target_temp)[
                    1:]  # Is this true? check if output_tokens == prompt_tokens - prefix_tokens
    output_length = len(output_tokens)
    output_labels = torch.tensor(output_tokens).to('cuda')  # shape = [label_len]
    # input = make_inputs(mt.tokenizer, [prompt])

    input = make_inputs(tokenizer, [prompt])
    logits = model(**input)["logits"]  # figure it out, shape = [bs, seq_len, vocab_size]
    logits = logits[0]  # shape = [seq_len, vocab_size]
    assert logits.shape[0] == len(prompt_tokens)
    shift_logits = logits[len(prefix_tokens) - 1: len(prompt_tokens) - 1, :]  # shift - 1
    assert shift_logits.shape[0] == output_length
    loss_fct = CrossEntropyLoss()
    ppl = loss_fct(shift_logits, output_labels).cpu().float()