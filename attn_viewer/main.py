import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import concurrent.futures
import os
import json
import pickle

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from matplotlib.colors import LinearSegmentedColormap
from accelerate import init_empty_weights, dispatch_model, load_checkpoint_and_dispatch

from interset_count import interset_count_nton, unionset_count_nton


def draw(similarity_result, layer_step, head_step):
    # similarity_result.shape = num_layer, num_heads, seqlen, seqlen
    num_layers, num_heads, seqlen, seqlen = similarity_result.shape
    similarity_result = similarity_result.cpu()

    for i in range(num_layers):
        fig, axes = plt.subplots(1, num_heads, figsize=(8 * num_heads, 6))
        layer_id = i * layer_step
        for j in range(num_heads):
            head_id = j * head_step

            print("             ", end="\r")
            print(i, j, end="\r")
            ax = axes[j]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

            sns.heatmap(similarity_result[i, j].numpy(), annot=False, fmt=".2f", cmap=cmap, ax=ax, vmin=0, vmax=1)
            ax.set_title(f'Heatmap layer{layer_id} head{head_id}')
            ax.invert_yaxis()

        plt.tight_layout()
        # plt.title(f'Heatmap layer{layer_id}')
        # plt.show()
        plt.savefig(f"heatmap_layer{layer_id}.png")

def similarity(tensor1, tensor2, topk):
    scores = interset_count_nton(tensor1, tensor2)
    max_num = tensor1.size(0) + topk
    union_count = unionset_count_nton(tensor1, tensor2, max_num)

    return scores / union_count
    # return scores / topk

def layer_query_similarity(layer_attn, topk, head_step):
    # layer_attn.shape = (bs, num_heads, seq_len, seq_len)
    bs, num_heads, seqlen, _ = layer_attn.shape
    assert bs == 1

    result = torch.zeros((num_heads // head_step, seqlen, seqlen))
    indices = torch.topk(layer_attn, topk, dim=-1).indices
    # TODO: debug
    # bs, num_heads = 1, 1

    for bs_idx in range(bs):
        for head_idx in range(0, num_heads, head_step):
            hi = head_idx // head_step

            scores = similarity(indices[bs_idx, head_idx], indices[bs_idx, head_idx], topk)

            scores = scores
            result[hi] = scores


            # for seqlen_idx1 in range(seqlen):
            #     for seqlen_idx2 in range(seqlen):
            #         if seqlen_idx2 > seqlen_idx1:
            #             break
            #         score = similarity(indices[bs_idx, head_idx, seqlen_idx1], indices[bs_idx, head_idx, seqlen_idx2])

            #         if seqlen_idx2 == seqlen_idx1:
            #             assert score == 1

            #         print(head_idx, seqlen_idx1, seqlen_idx2, end="\r")
            #         result[head_idx, seqlen_idx1, seqlen_idx2] = score
            #         result[head_idx, seqlen_idx2, seqlen_idx1] = score


    return result

def load_jsonl(data_path):
    datas = []
    if os.path.exists(data_path):
        f = open(data_path, 'r', encoding='utf-8')
        for line in f.readlines():
            datas.append(json.loads(line))
    else:
        print(f"not exists: {data_path}")
    return datas


def main(args):
    print(args.model_name)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation='eager', torch_dtype=torch.bfloat16, device_map="auto", offload_folder="offload", use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # prompt = "The quick brown fox jumps over the lazy dog" * 128
    prompt_format = "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:"
    datas = load_jsonl("./loogle_SD_mixup_128k_1.jsonl")
    json_data = datas[0]
    prompt = prompt_format.format(**json_data)


    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    # # TODO: debug
    # input_ids = input_ids[:, :1024 * 4]
    input_ids = input_ids[:, :1024]
    seqlen = input_ids.size(1)
    print(input_ids.shape)

    with torch.no_grad():
        # NOTE: [layer_attn, ...]
        attention_scores = model(input_ids, output_attentions=True, use_cache=False)['attentions']

    # print("dumping attention scores...")
    # for layer_id, layer_attn in tqdm(enumerate(attention_scores)):
    #     pickle.dump(layer_attn.cpu(), open(f"attn_layer_{layer_id}_seq{seqlen}.pkl", "wb"))
    # print("dumping attention scores done.")

    # TODO: mask out last block
    # print(attention_scores[0].shape)
    # print(attention_scores[0])

    topk = 512
    head_step = 4
    layer_step = 4

    layers_similarity = []
    for layer_id, layer_attn in tqdm(enumerate(attention_scores)):
        if not layer_id % layer_step == 0:
            continue

        # TODO: trim prefix tokens
        trim = topk
        layer_attn = layer_attn[:, :, trim:, :]
        layer_result = layer_query_similarity(layer_attn, topk, head_step)
        pickle.dump(layer_result.cpu(), open(f"sim_layer_{layer_id}_seq{seqlen}_top{topk}.pkl", "wb"))
        layers_similarity.append(layer_result)


    layers_similarity = torch.stack(layers_similarity)
    # print(layers_similarity.shape)

    draw(layers_similarity, layer_step, head_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, required=True)
    args = parser.parse_args()
    main(args)
