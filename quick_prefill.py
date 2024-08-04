import torch

from kv_cache import QuestCache

CACHE_BLOCK = 128
PREFILL_BLOCK = 1024
TOPK = 4
BUDGETS = CACHE_BLOCK * TOPK

@torch.no_grad()
def generate(input_ids, model, max_new_tokens=50, eos_token_id=[]):
    next_input = input_ids
    generated_ids = input_ids
    max_length = input_ids.size(-1) + max_new_tokens

    past_key_values = None
    while generated_ids.size(-1) < max_length:
        outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        next_input = next_token_id
        past_key_values = outputs.past_key_values

        if next_input[0] in eos_token_id:
            break

    return generated_ids

def segment_prefill(input_ids, model, prefill_block_size=PREFILL_BLOCK, cache_block_size=CACHE_BLOCK, budget_size=BUDGETS):
    input_len = input_ids.size(-1)
    past_key_values = None

    kv_cache = QuestCache(cache_block_size, skip_layers=1, budget_size=budget_size, model=model)
    threshold_len = prefill_block_size

    for i in range(0, input_len, prefill_block_size):
        input_segment = input_ids[:, i:min(i+prefill_block_size, input_len)]

        outputs = model(input_segment, past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values
        kv_cache.update(past_key_values, new_cache_len=input_segment.size(-1))

        if i > threshold_len and i+prefill_block_size+1 < input_len:
            # TODO: single query to snap for now?
            query = input_ids[:, i+prefill_block_size+1:i+prefill_block_size+2]
            # use compressed cache

            past_key_values = kv_cache(past_key_values, query=query)

    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    past_key_values = kv_cache.past_key_values
    # TODO:
    assert input_len == past_key_values[-1][0].size(-2)

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
            next_input, past_key_values = segment_prefill(next_input, model, prefill_block_size=PREFILL_BLOCK)
        else:
            outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_input = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
        generated_ids = torch.cat([generated_ids, next_input], dim=-1)

        # TODO: single batch for now
        if next_input[0] in eos_token_id:
            break

    return generated_ids


