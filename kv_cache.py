import torch
import random
import numpy as np
from modeling_patch import replace_naive_attention, replace_flash_attention

def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]

import torch.nn.functional as F
DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

class QuestCache:
    def __init__(
        self,
        block_size,
        skip_layers,
        ratio,
        model,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        self.block_size = block_size
        self.skip_layers = skip_layers
        self.ratio = ratio
        self.model = model

        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.past_key_values = []

    def __call__(self, past_key_values, num_of_token=None, attentions=None, query=None, new_cache_len=1):
        # update local kv cache
        if len(self.past_key_values) == 0:
            # TODO: slow update
            for layer_idx, (k, v) in enumerate(past_key_values):
                self.past_key_values.append([k, v])
        else:
            for layer_idx, (k, v) in enumerate(past_key_values):
                self.past_key_values[layer_idx][0] = torch.cat([self.past_key_values[layer_idx][0], k[:,:,-new_cache_len:,:]], dim=self.k_seq_dim)
                self.past_key_values[layer_idx][1] = torch.cat([self.past_key_values[layer_idx][1], v[:,:,-new_cache_len:,:]], dim=self.v_seq_dim)

        past_key_values = self.past_key_values

        # NOTE: profiling with block max/min
        bs, head_num, seq_len, head_dim = past_key_values[0][0].shape
        max_block_num = (seq_len + self.block_size - 1) // self.block_size
        device=past_key_values[0][0].device

        # estimated_block
        block_max = []
        block_min = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            layer_block_max_k = []
            layer_block_max_v = []
            layer_block_min_k = []
            layer_block_min_v = []
            for i in range(0, seq_len, self.block_size):
                start_index = i
                end_index = start_index + self.block_size

                layer_block_max_k.append(k[:, :, start_index:end_index].max(dim=-2, keepdim=True).values)
                layer_block_max_v.append(v[:, :, start_index:end_index].max(dim=-2, keepdim=True).values)
                layer_block_min_k.append(k[:, :, start_index:end_index].min(dim=-2, keepdim=True).values)
                layer_block_min_v.append(v[:, :, start_index:end_index].min(dim=-2, keepdim=True).values)

            layer_block_max_k = torch.cat(layer_block_max_k, dim=self.k_seq_dim)
            layer_block_max_v = torch.cat(layer_block_max_v, dim=self.v_seq_dim)
            layer_block_min_k = torch.cat(layer_block_min_k, dim=self.k_seq_dim)
            layer_block_min_v = torch.cat(layer_block_min_v, dim=self.v_seq_dim)
            block_max.append([layer_block_max_k, layer_block_max_v])
            block_min.append([layer_block_min_k, layer_block_min_v])

        # attn_score.shape = bs, num_heads, qlen, klen
        replace_naive_attention(self.model)
        max_attentions = self.model(
            query,
            past_key_values=block_max,
            use_cache=True,
            output_attentions=True,
        ).attentions
        min_attentions = self.model(
            query,
            past_key_values=block_min,
            use_cache=True,
            output_attentions=True,
        ).attentions
        replace_flash_attention(self.model)

        selected_cache = []
        topk = int((num_of_token * self.ratio) / self.block_size)
        for layer_idx in range(len(max_attentions)):
            # remove last token (a snap query)
            max_attention = max_attentions[layer_idx].mean(dim=-2)[:,:,:-2]
            min_attention = min_attentions[layer_idx].mean(dim=-2)[:,:,:-2]

            channel_max = torch.max(max_attention, min_attention)
            # TODO: hard code for now
            # block index
            # TODO: sort? keep pos
            index = torch.topk(channel_max, topk, dim=-1).indices.sort(dim=-1).values

            k, v = past_key_values[layer_idx]
            # select block tokens
            # TODO: naive implementation
            layer_selected_k = []
            layer_selected_v = []

            # TODO: batch select

            # algo 1
            # padding
            # reshape (bs, num_heads, qlen // block_size, block_size, dim)
            # selecet
            r = torch.arange(0, max_block_num * self.block_size, device=device)[None, None, :]
            r = r.expand(bs, head_num, max_block_num * self.block_size).view(bs, head_num, max_block_num, self.block_size)

            index = torch.gather(r, 2, index.unsqueeze(-1).expand(-1, -1, -1, self.block_size))
            index = index.view(bs, head_num, -1)

            layer_selected_k = torch.gather(k, dim=-2, index=index.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            layer_selected_v = torch.gather(v, dim=-2, index=index.unsqueeze(-1).expand(-1, -1, -1, head_dim))

            selected_cache.append([layer_selected_k, layer_selected_v])

        return selected_cache

