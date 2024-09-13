import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    repeat_kv,
)

def cache_profilling(fwd_obj, query_states, key_states, block_size, budget_size, segment_size):
    '''
    q,k (bs, seqlen, num_heads, head_dim)
    return index shape = (bs, num_heads, num_segments, topk)
    '''

    assert segment_size >= block_size

    # (bs, seqlen, num_heads, head_dim)
    bsz, seqlen, num_heads, head_dim = query_states.shape
    num_heads_k = key_states.size(-2)
    num_key_value_groups = num_heads // num_heads_k
    nrep = segment_size // block_size

    num_segments = (seqlen + segment_size - 1) // segment_size
    pool_len = (num_segments - 1) * segment_size
    num_blocks = pool_len // block_size
    num_segments = pool_len // segment_size
    query_states = query_states[:, :pool_len]
    key_states = key_states[:, :pool_len]

    query_states = query_states.transpose(1, 2).reshape(bsz, num_heads, pool_len//segment_size, segment_size, head_dim)

    # NOTE: GQA support
    key_states = key_states.transpose(1, 2)
    key_states = repeat_kv(key_states, num_key_value_groups)
    key_states = key_states.reshape(bsz, num_heads, pool_len//block_size, block_size, head_dim)

    # (bs, num_heads, block_num, head_dim)
    layer_max_q = query_states.max(dim=-2).values
    layer_max_k = key_states.max(dim=-2).values
    layer_min_q = query_states.min(dim=-2).values
    layer_min_k = key_states.min(dim=-2).values

    # (bs, num_heads, seg_num_q, block_num_k)
    q_block_len = layer_max_q.size(-2)
    k_block_len = layer_max_k.size(-2)
    qq = torch.cat([layer_max_q, layer_min_q], dim=-2)
    attn_weights_max = torch.matmul(qq, layer_max_k.transpose(2, 3)).view(bsz, num_heads, 2, q_block_len, k_block_len).mean(dim=2)
    attn_weights_min = torch.matmul(qq, layer_min_k.transpose(2, 3)).view(bsz, num_heads, 2, q_block_len, k_block_len).mean(dim=2)
    attn_weights = torch.max(attn_weights_max, attn_weights_min)

    # GQA support
    attn_weights = attn_weights.view(bsz, num_heads_k, num_key_value_groups, num_segments, num_blocks).mean(dim=2)

    mask = torch.full((num_blocks, num_blocks), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)

    # NOTE: always keep the last segment, to keep first layers's accuracy and let not kv behind
    mask.masked_fill_(mask_cond + nrep < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device).view(num_segments, nrep, num_blocks)[:, -1, :]
    attn_weights += mask[None, None, :, :]

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(key_states.dtype)

    if fwd_obj.layer_fusion == True:
        if not hasattr(fwd_obj._g, "prev_attn_weights") or fwd_obj.layer_idx <= 1:
            fwd_obj._g.prev_attn_weights = attn_weights
        else:
            pprev = 0.25
            pcurr = 1 - pprev
            cdevice = attn_weights.device
            attn_weights = fwd_obj._g.prev_attn_weights.to(cdevice) * pprev + attn_weights * pcurr
            fwd_obj._g.prev_attn_weights = attn_weights

    topk = budget_size // block_size
    index = torch.topk(attn_weights, topk, dim=-1).indices

    return index

def cache_selection(key_states, value_states, index, block_size, left_over):
    '''
    key_states.shape = (bs, seqlen, num_heads, head_dim)
    index.shape = (bs, num_heads, topk)
    '''
    bsz, seqlen, num_head, head_dim = key_states.shape

    index = index.transpose(1, 2).unsqueeze(2).unsqueeze(-1).expand(bsz, -1, block_size, num_head , head_dim)

    selected_k = torch.gather(key_states[:, :-left_over].view(bsz, -1, block_size, num_head, head_dim), dim=1, index=index)
    selected_v = torch.gather(value_states[:, :-left_over].view(bsz, -1, block_size, num_head , head_dim), dim=1, index=index)

    selected_k = selected_k.view(bsz, -1, num_head, head_dim)
    selected_v = selected_v.view(bsz, -1, num_head, head_dim)

    layer_selected_k = torch.cat([selected_k, key_states[:, -left_over:]], dim=1)
    layer_selected_v = torch.cat([selected_v, value_states[:, -left_over:]], dim=1)
    return layer_selected_k, layer_selected_v


def eattention(fwd_obj, segment_size, threshold_len, budgets, block_size, query_states, key_states, value_states):
    from flash_attn import flash_attn_func
    # NOTE: q,k,v shape = (bsz, len, num_heads, head_dim)

    # index.shape = (bs, num_heads, num_segments, topk)
    block_index = cache_profilling(fwd_obj, query_states, key_states, block_size, budgets, segment_size)

    input_len = query_states.size(1)
    if input_len == 1:
        assert False, "not support for now"
        # decoding
    else:
        # NOTE: full cache k, v
        attn_outputs = []
        for i in range(0, input_len, segment_size):
            q_segment = query_states[:, i:i+segment_size,:,:]
            k_segment = key_states[:, :i+segment_size,:,:]
            v_segment = value_states[:, :i+segment_size,:,:]

            if i >= threshold_len and i + segment_size < input_len:
                # NOTE: keep last block full len
                curr_segment = i // segment_size
                index = block_index[:, :, curr_segment, :]
                k_segment, v_segment = cache_selection(k_segment, v_segment, index, block_size, segment_size)

            attn_output = flash_attn_func(
                q_segment, k_segment, v_segment, causal=True
            )

            attn_outputs.append(attn_output)

        attn_output = torch.cat(attn_outputs, dim=1)
    return attn_output

