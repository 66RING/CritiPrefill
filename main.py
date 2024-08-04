import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

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



def main(args):
    device = "auto"
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                                 device_map=device,
                                                 torch_dtype=dtype,
                                                 attn_implementation="flash_attention_2"
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, device_map=device)
    prompt = "Once upon a time."

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated_ids = generate(input_ids, model, tokenizer)
    generated_text = tokenizer.decode(generated_ids[0, input_ids.size(-1):], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, default="gpt2")
    args = parser.parse_args()
    main(args)
