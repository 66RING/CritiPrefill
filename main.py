import torch
import argparse
from kv_cache import QuestCache
from transformers import AutoTokenizer
# from modeling_llama import LlamaForCausalLM
from transformers import LlamaForCausalLM
from modeling_patch import replace_llama
import time

replace_llama()

@torch.no_grad()
def generate(input_ids, model, tokenizer, max_generation=50):
    next_input = input_ids
    generated_ids = input_ids
    max_length = input_ids.size(-1) + max_generation

    past_key_values = None
    while generated_ids.size(-1) < max_length:
        outputs = model(next_input, past_key_values=past_key_values,)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        next_input = next_token_id
        past_key_values = outputs.past_key_values
    return generated_ids

def segment_prefill(input_ids, model, block_size):
    input_len = input_ids.size(-1)
    past_key_values = None

    kv_cache = QuestCache(block_size, skip_layers=0, ratio=0.5, model=model)
    threshold_len = block_size

    for i in range(0, input_len, block_size):
        input_segment = input_ids[:, i:i+block_size]
        outputs = model(input_segment, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values

        if i > threshold_len and i+block_size+1 < input_len:
            # TODO: single query to snap for now?
            query = input_ids[:, i+block_size+1:i+block_size+2]
            past_key_values = kv_cache(past_key_values, new_cache_len=block_size, num_of_token=i+block_size,attentions=None, query=query)

    return past_key_values

@torch.no_grad()
def quick_generate(input_ids, model, tokenizer, max_generation=50):
    next_input = input_ids
    generated_ids = input_ids
    max_length = input_ids.size(-1) + max_generation


    past_key_values = None
    while generated_ids.size(-1) < max_length:
        if past_key_values is None:
            past_key_values = segment_prefill(next_input, model, block_size=4)
        else:
            outputs = model(next_input, past_key_values=past_key_values,)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            next_input = next_token_id
            past_key_values = outputs.past_key_values

    return generated_ids



def main(args):
    device = "auto"
    dtype = torch.bfloat16
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, 
                                                 device_map=device,
                                                 torch_dtype=dtype,
                                                 attn_implementation="flash_attention_2"
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, device_map=device)
    # prompt = "Once upon a time."
    prompt = "One day, Lily met a Shoggoth." * 128
    prompt = "Once upon a time. One day, Lily met a Shoggoth and a dragon."

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    t = time.time()
    torch.cuda.synchronize()
    generated_ids = generate(input_ids, model, tokenizer)
    torch.cuda.synchronize()
    t = time.time() - t
    print("naive time:", t)

    generated_text = tokenizer.decode(generated_ids[0, input_ids.size(-1):], skip_special_tokens=True)
    print("naive>", generated_text)


    t = time.time()
    torch.cuda.synchronize()
    generated_ids = quick_generate(input_ids, model, tokenizer)
    torch.cuda.synchronize()
    t = time.time() - t
    print("quick time:", t)

    generated_text = tokenizer.decode(generated_ids[0, input_ids.size(-1):], skip_special_tokens=True)
    print("seg>", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, default="gpt2")
    args = parser.parse_args()
    main(args)
