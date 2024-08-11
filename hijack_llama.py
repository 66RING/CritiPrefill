import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math

from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

class ECache:
    def __init__(self, bsz, num_heads, head_dim, device, dtype):
        self.max_key_cache = torch.empty((bsz, num_heads, 0, head_dim), device=device, dtype=dtype)
        self.min_key_cache = torch.empty((bsz, num_heads, 0, head_dim), device=device, dtype=dtype)

    def update(self, key_states, block_size):
        '''
        key_states shape = (bsz, len, num_heads, head_dim)

        return shape = (bsz, num_heads, len, head_dim)
        '''
        bsz, _, head_num, head_dim = key_states.shape
        key_states = key_states.transpose(1, 2)
        key_len = key_states.size(2)
        key_candidate_len = key_len // block_size
        if key_candidate_len <= self.max_key_cache.size(2):
            return self.max_key_cache, self.min_key_cache

        pooling = nn.MaxPool1d(block_size, stride=block_size)
        new_cache_len = int((key_candidate_len - self.max_key_cache.size(2)) * block_size)
        key_states = key_states[:,:,-new_cache_len:].view(bsz * head_num, -1, head_dim).transpose(-1, -2)
        new_max_key_cache = pooling(key_states)
        new_min_key_cache = pooling(key_states * -1) * -1
        new_max_key_cache = new_max_key_cache.transpose(-1, -2).view(bsz, head_num, -1, head_dim)
        new_min_key_cache = new_min_key_cache.transpose(-1, -2).view(bsz, head_num, -1, head_dim)
        self.max_key_cache = torch.cat([self.max_key_cache, new_max_key_cache], dim=2)
        self.min_key_cache = torch.cat([self.min_key_cache, new_min_key_cache], dim=2)
        return self.max_key_cache, self.min_key_cache

def llama_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if hasattr(self, "naive") and self.naive:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    else:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def llama_eattention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    import nvtx
    import os
    def profiling(future_query_states, key_states, value_states):
        import wandb
        import numpy as np
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # (bs, head, qlen, klen) => (head, klen)
        qk = torch.matmul(future_query_states, key_states.transpose(2, 3))
        attn_score = qk.mean(dim=-2).mean(dim=0)

        # (bs, head, klen, dim) => (head, klen)
        v_mean_score = value_states.mean(dim=-1).mean(dim=0)

        # (bs, head, klen, dim) => (head, klen)
        v_max_score = value_states.max(dim=-1).values.mean(dim=0)
        
        v_min_score = value_states.min(dim=-1).values.mean(dim=0)

        assert attn_score.size() == v_mean_score.size() == v_max_score.size() == v_min_score.size(), f"{attn_score.shape}, {v_mean_score.shape}, {v_max_score.shape}"
        for hi in range(attn_score.size(0)):
            # for topk in [1024, 512, 256, 128, 64, 32, 16, 8]:
            for topk in [8]:
                base_topk = attn_score[hi].topk(topk).indices.sort().values.cpu()
                v_mean_topk = v_mean_score[hi].topk(topk).indices.sort().values.cpu()
                v_max_topk = v_max_score[hi].topk(topk).indices.sort().values.cpu()
                v_min_topk = v_min_score[hi].topk(topk).indices.sort().values.cpu()
                print(base_topk)
                print(len(np.intersect1d(v_mean_topk, base_topk)), v_mean_topk)
                print(len(np.intersect1d(v_max_topk, base_topk)), v_max_topk)
                print(len(np.intersect1d(v_min_topk, base_topk)), v_min_topk)
                # print(len(np.intersect1d(v_mean_topk, base_topk)))
                # print(len(np.intersect1d(v_max_topk, base_topk)))
                # print(len(np.intersect1d(v_min_topk, base_topk)))
                input()

                pass


    def cache_compress(future_query_states, key_states, value_states, cache_len, block_size, budget_size, stay_size):
        # copy_old_rng = nvtx.start_range("copy old")
        # nvtx.end_range(copy_old_rng)


        # return key_states[:, :budget_size], value_states[:, :budget_size]
        # q,k,v shape = (bsz, len, num_heads, head_dim)
        assert block_size <= stay_size

        bsz, future_q_len, head_num, head_dim = future_query_states.shape
        device = future_query_states.device
        layer_block_max_k = []
        layer_block_min_k = []
        left_over = 0
        
        # # TODO: algo 1
        # # TODO: algo2: pooling and then sampling
        # # NOTE: always keep last segment to compute new cache
        # cnt = 0
        # for i in range(0, cache_len - stay_size, block_size):
        #     print(i)
        #     cnt += 1
        #     start_index = i
        #     end_index = start_index + block_size
        #     left_over = cache_len - end_index

        #     layer_block_max_k.append(key_states[:, start_index:end_index].max(dim=1, keepdim=True).values)
        #     layer_block_min_k.append(key_states[:, start_index:end_index].min(dim=1, keepdim=True).values)


        # # algo 2
        # layer_block_max_k = key_states[:,:(cache_len - stay_size)//block_size]
        # layer_block_min_k = key_states[:,:(cache_len - stay_size)//block_size]
        # left_over = cache_len - (cache_len - stay_size)//block_size * block_size

        # future_query_states = future_query_states.transpose(1, 2)
        # layer_block_max_k = layer_block_max_k.transpose(1, 2)
        # layer_block_min_k = layer_block_min_k.transpose(1, 2)
        # # NOTE: here q,k,v shape = (bsz, num_heads, len, head_dim)


        # # algo 3 pooling
        # future_query_states = future_query_states.transpose(1, 2)
        # layer_block_max_k = key_states.transpose(1, 2)
        # layer_block_min_k = layer_block_max_k
        # import torch.nn
        # pool_len = (cache_len - stay_size)//block_size * block_size
        # pooling = nn.MaxPool1d(block_size, stride=block_size)
        # layer_block_max_k = pooling(layer_block_max_k[:,:,:pool_len].view(bsz * head_num, -1, head_dim).transpose(-1, -2))
        # layer_block_min_k = pooling(layer_block_min_k[:,:,:pool_len].view(bsz * head_num, -1, head_dim).transpose(-1, -2) * -1) * -1
        # left_over = cache_len - pool_len
        # layer_block_max_k = layer_block_max_k.transpose(-1, -2).view(bsz, head_num, -1, head_dim)
        # layer_block_min_k = layer_block_min_k.transpose(-1, -2).view(bsz, head_num, -1, head_dim)


        # algo 4: streaming, 最后一个block始终保留, 之前的block使用缓存
        future_query_states = future_query_states.transpose(1, 2)
        # # NOTE: here q,k,v shape = (bsz, len, num_heads, head_dim)
        pool_len = (cache_len - stay_size)//block_size * block_size
        left_over = cache_len - pool_len
        layer_block_max_k, layer_block_min_k = self.eattn_cache.update(key_states[:, :pool_len], block_size)
        assert layer_block_max_k.size(2)  * block_size == pool_len

        # shape = (bsz, num_heads, future_q_len, block_nums)
        attn_weights_max = torch.matmul(future_query_states, layer_block_max_k.transpose(2, 3))
        attn_weights_min = torch.matmul(future_query_states, layer_block_min_k.transpose(2, 3))

        # shape = (bsz, num_heads, block_nums)
        # NOTE: max/min pooling along the future_q_len
        attn_weights_max = attn_weights_max.max(dim=2).values
        attn_weights_min = attn_weights_min.max(dim=2).values
        # attn_weights_max = attn_weights_max.mean(dim=2)
        # attn_weights_min = attn_weights_min.mean(dim=2)
        channel_max = torch.max(attn_weights_min, attn_weights_max)

        # TODO: -1
        topk = budget_size // block_size - 1
        index = torch.topk(channel_max, topk, dim=-1).indices

        # TODO: always keep last segment?? and sink token?

        # profiling(future_query_states, key_states, value_states)

        def block_selection(key_states, value_states, index):
            # # index.shape = (bsz, num_heads, block_nums)
            # # key_states.shape = (bsz, len, num_heads, head_dim)
            # key_states = key_states.transpose(1, 2).view(bsz, head_num, -1, block_size, head_dim)
            # value_states = value_states.transpose(1, 2).view(bsz, head_num, -1, block_size, head_dim)
            # index = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, block_size, head_dim)

            # selected_k = torch.gather(key_states, dim=2, index=index)
            # selected_v = torch.gather(value_states, dim=2, index=index)
            # selected_k = selected_k.view(bsz, head_num, -1, head_dim).transpose(1, 2)
            # selected_v = selected_v.view(bsz, head_num, -1, head_dim).transpose(1, 2)
            # return selected_k, selected_v

            key_states = key_states.view(bsz, -1, block_size, head_num, head_dim)
            value_states = value_states.view(bsz, -1, block_size, head_num , head_dim)
            index = index.transpose(1, 2).unsqueeze(2).unsqueeze(-1).expand(bsz, -1, block_size, head_num , head_dim)

            selected_k = torch.gather(key_states, dim=1, index=index)
            selected_v = torch.gather(value_states, dim=1, index=index)
            selected_k = selected_k.view(bsz, -1, head_num, head_dim)
            selected_v = selected_v.view(bsz, -1, head_num, head_dim)
            return selected_k, selected_v

        layer_selected_k, layer_selected_v = block_selection(key_states[:, :pool_len], value_states[:, :pool_len], index)

        layer_selected_k = torch.cat([layer_selected_k, key_states[:, -left_over:]], dim=1)
        layer_selected_v = torch.cat([layer_selected_v, value_states[:, -left_over:]], dim=1)
        return layer_selected_k, layer_selected_v


    def eattention(segment_size, threshold_len, budgets, block_size, query_states, key_states, value_states):
        from flash_attn import flash_attn_func
        # NOTE: q,k,v shape = (bsz, len, num_heads, head_dim)

        input_len = query_states.size(1)
        if input_len == 1:
            # decoding
            # TODO: decoding bug
            cache_len = key_states.size(1)
            key_states, value_states = cache_compress(query_states, key_states, value_states, cache_len, block_size, budgets, block_size)
            attn_output = flash_attn_func(
                query_states, key_states, value_states, causal=True
            )
        else:
            # NOTE: full cache k, v
            attn_outputs = []
            for i in range(0, input_len, segment_size):
                q_segment = query_states[:, i:i+segment_size,:,:]
                k_segment = key_states[:, :i+segment_size,:,:]
                v_segment = value_states[:, :i+segment_size,:,:]
                if i > threshold_len or input_len == 1:
                    future_q = q_segment[:, 0::segment_size//10, :]
                    cache_len = k_segment.size(1)
                    k_segment, v_segment = cache_compress(future_q, k_segment, v_segment, cache_len, block_size, budgets, segment_size)

                attn_output = flash_attn_func(
                    q_segment, k_segment, v_segment, causal=True
                )
                attn_outputs.append(attn_output)

            attn_output = torch.cat(attn_outputs, dim=1)
        return attn_output


    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    # NOTE: hard code for now
    is_prefill = q_len != 1

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # init lookup table
    if is_prefill:
        self.eattn_cache = ECache(bsz, self.num_heads, self.head_dim, device=query_states.device, dtype=query_states.dtype)

    prefill_only = os.environ.get('PREFILL_ONLY', '0') == '1'
    do_eattn = True
    if prefill_only and not is_prefill:
        do_eattn = False

    # if is_prefill and self.layer_idx > 0:
    if self.layer_idx > 0 and do_eattn:
        # TODO: hard code for now
        segment_size = int(os.environ.get('SEG_SIZE', '4096'))
        threshold_len = int(os.environ.get('SEG_START', '4096'))
        block_size = int(os.environ.get('BLOCK_SIZE', '32'))
        budgets = int(os.environ.get('BUDGETS', '512'))

        attn_output = eattention(segment_size, threshold_len, budgets, block_size, query_states, key_states, value_states)
    else:
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )
    assert attn_output.size(1) == q_len, f"outlen={attn_output.size(1)}, q_len={q_len}"

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


