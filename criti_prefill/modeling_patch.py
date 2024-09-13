import transformers
from criti_prefill.hijack_llama import llama_eattention_forward, LlamaModel_forward

from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)

def replace_llama_eattention():
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_eattention_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward

def replace_base():
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward

def criti_config(model, segment_size=512, threshold_len=4096, block_size=32, budgets=2048, prefill_only=True, layer_fusion=True, layer_skip=1):

    for layer in model.model.layers:
        layer.self_attn.segment_size = segment_size
        layer.self_attn.threshold_len = threshold_len
        layer.self_attn.block_size = block_size
        layer.self_attn.budgets = budgets
        layer.self_attn.prefill_only = prefill_only
        layer.self_attn.layer_fusion = layer_fusion
        layer.self_attn.layer_skip = layer_skip

    print(f"Criti config: segment_size={segment_size}, threshold_len={threshold_len}, block_size={block_size}, budgets={budgets}, prefill_only={prefill_only}, layer_fusion={layer_fusion}, layer_skip={layer_skip}", flush=True)

