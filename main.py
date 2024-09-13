import torch
import argparse
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
# from modeling_llama import LlamaForCausalLM
from transformers import LlamaForCausalLM
from criti_prefill.modeling_patch import replace_llama_eattention, criti_config

def main(args):
    if args.enable_eattention:
        replace_llama_eattention()
    torch.manual_seed(0)
    device = "auto"
    dtype = torch.bfloat16
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, 
                                                 device_map=device,
                                                 torch_dtype=dtype,
                                                 attn_implementation="flash_attention_2"
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, device_map=device)
    # prompt = "Once upon a time."
    prompt = "One day, Lily met a Shoggoth." * 1024 * 2
    # prompt = "Once upon a time. One day, Lily met a Shoggoth and a dragon."

    criti_config(model,
                 segment_size=args.segment_size,
                 threshold_len=args.threshold_len,
                 block_size=args.block_size,
                 budgets=args.budgets,
                 layer_fusion=args.layer_fusion,
                 layer_skip=args.layer_skip)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    MAX_GEN_LENGTH = 50

    t = time.time()
    torch.cuda.synchronize()
    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_GEN_LENGTH,
        use_cache=True,
        return_dict_in_generate=True).sequences
    torch.cuda.synchronize()
    t = time.time() - t
    print("time:", t)
    generated_text = tokenizer.decode(generated_ids[0, input_ids.size(-1):], skip_special_tokens=True)
    print(">", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("-e", "--enable_eattention", action='store_true')
    parser.add_argument("--segment_size", type=int, default=512)
    parser.add_argument("--threshold_len", type=int, default=1024)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budgets", type=int, default=2048)
    parser.add_argument("--layer_skip", type=int, default=1)
    parser.add_argument("--layer_fusion", action='store_true')
    args = parser.parse_args()
    main(args)


