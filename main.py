import torch
import argparse
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
# from modeling_llama import LlamaForCausalLM
from transformers import LlamaForCausalLM
from modeling_patch import replace_llama
from quick_prefill import generate, quick_generate

replace_llama()

def main(args):
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

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    t = time.time()
    torch.cuda.synchronize()
    generated_ids = generate(input_ids, model)
    torch.cuda.synchronize()
    t = time.time() - t
    print("naive time:", t)

    generated_text = tokenizer.decode(generated_ids[0, input_ids.size(-1):], skip_special_tokens=True)
    print("nai>", generated_text)


    t = time.time()
    torch.cuda.synchronize()
    generated_ids = quick_generate(input_ids, model)
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
