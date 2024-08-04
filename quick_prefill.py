import torch

from kv_cache import QuestCache

BLOCK_SIZE = 128
RATIO = 0.5

@torch.no_grad()
def generate(input_ids, model, max_new_tokens=50, eos_token_id=[]):
    next_input = input_ids
    generated_ids = input_ids
    max_length = input_ids.size(-1) + max_new_tokens

    past_key_values = None
    # TODO: remove
    is_prefill = True
    while generated_ids.size(-1) < max_length:
        outputs = model(next_input, past_key_values=past_key_values,)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        next_input = next_token_id
        past_key_values = outputs.past_key_values

        if is_prefill:
            print("naive prefill cache.shape", past_key_values[0][0].shape)
        is_prefill = False

        if next_input[0] in eos_token_id:
            break

    print("naive", past_key_values[0][0].shape)
    return generated_ids

def segment_prefill(input_ids, model, block_size):
    input_len = input_ids.size(-1)
    past_key_values = None

    kv_cache = QuestCache(block_size, skip_layers=1, ratio=RATIO, model=model)
    threshold_len = block_size

    for i in range(0, input_len, block_size):
        input_segment = input_ids[:, i:i+block_size]
        outputs = model(input_segment, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values

        if i > threshold_len and i+block_size+1 < input_len:
            # TODO: single query to snap for now?
            query = input_ids[:, i+block_size+1:i+block_size+2]
            # use compressed cache
            past_key_values = kv_cache(past_key_values, new_cache_len=block_size, num_of_token=min(i+block_size, input_len), attentions=None, query=query)
        else:
            query = input_ids[:, i+block_size+1:i+block_size+2]
            # record history cache
            # TODO: duplicate kv_cache occupy?
            kv_cache(past_key_values, new_cache_len=block_size, num_of_token=min(i+block_size, input_len), attentions=None, query=query)

    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    return next_token_id, past_key_values

@torch.no_grad()
def quick_generate(input_ids, model, max_new_tokens=50, eos_token_id=[]):
    assert input_ids.size(0) == 1
    next_input = input_ids
    generated_ids = input_ids
    max_length = input_ids.size(-1) + max_new_tokens

    past_key_values = None
    while generated_ids.size(-1) < max_length:
        if past_key_values is None:
            next_input, past_key_values = segment_prefill(next_input, model, block_size=BLOCK_SIZE)
            print("prefill cache.shape", past_key_values[0][0].shape)
        else:
            outputs = model(next_input, past_key_values=past_key_values,)
            next_token_logits = outputs.logits[:, -1, :]
            next_input = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
        generated_ids = torch.cat([generated_ids, next_input], dim=-1)

        # TODO: single batch for now
        if next_input[0] in eos_token_id:
            break
    print("quick", past_key_values[0][0].shape)


    return generated_ids


