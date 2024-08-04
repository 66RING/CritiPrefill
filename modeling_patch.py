import transformers
from hijack_llama import llama_attn_forward

def replace_llama():
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_attn_forward

def replace_naive_attention(model):
    for layer in model.model.layers:
        layer.self_attn.naive = True

def replace_flash_attention(model):
    for layer in model.model.layers:
        layer.self_attn.naive = False
