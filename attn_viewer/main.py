import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import concurrent.futures

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.colors import LinearSegmentedColormap


def draw(similarity_result):
    # similarity_result.shape = num_layer, num_heads, seqlen, seqlen
    num_layers, num_heads, seqlen, seqlen = similarity_result.shape
    similarity_result = similarity_result.cpu()

    for i in range(num_layers):
        fig, axes = plt.subplots(1, num_heads, figsize=(8 * num_heads, 6))
        for j in range(num_heads):
            print("             ", end="\r")
            print(i, j, end="\r")
            ax = axes[j]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

            sns.heatmap(similarity_result[i, j].numpy(), annot=False, fmt=".2f", cmap=cmap, ax=ax, vmin=0, vmax=1)
            ax.set_title(f'Heatmap layer{i}-head{j}')
            ax.invert_yaxis()

        plt.tight_layout()
        plt.title(f'Heatmap layer{i}')
        # plt.show()
        plt.savefig(f"heatmap_layer{i}.png")


def similarity(tensor1, tensor2):
    len = tensor1.numel()
    intersection = np.intersect1d(tensor1.cpu(), tensor2.cpu())

    intersection_size = intersection.size
    return intersection_size / len


def layer_query_similarity(layer_attn, topk):
    # layer_attn.shape = (bs, num_heads, seq_len, seq_len)
    bs, num_heads, seqlen, _ = layer_attn.shape
    assert bs == 1

    result = torch.zeros((num_heads, seqlen, seqlen))
    indices = torch.topk(layer_attn, topk, dim=-1).indices
    # TODO: debug
    # bs, num_heads = 1, 1

    for bs_idx in range(bs):
        for head_idx in range(num_heads):

            for seqlen_idx1 in range(seqlen):
                for seqlen_idx2 in range(seqlen):
                    if seqlen_idx2 > seqlen_idx1:
                        break
                    score = similarity(indices[bs_idx, head_idx, seqlen_idx1], indices[bs_idx, head_idx, seqlen_idx2])

                    if seqlen_idx2 == seqlen_idx1:
                        assert score == 1

                    # print(seqlen_idx1, seqlen_idx2, score, end="\r")
                    result[head_idx, seqlen_idx1, seqlen_idx2] = score
                    result[head_idx, seqlen_idx2, seqlen_idx1] = score


    return result


def main(args):
    print(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation='eager', torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    prompt = "The quick brown fox jumps over the lazy dog" * 128
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    # # TODO: debug
    # input_ids = input_ids[:, :128]
    print(input_ids.shape)

    with torch.no_grad():
        # NOTE: [layer_attn, ...]
        attention_scores = model(input_ids, output_attentions=True, use_cache=False)['attentions']

    # TODO: mask out last block
    # print(attention_scores[0].shape)
    # print(attention_scores[0])

    topk = 32
    layers_similarity = []
    for layer_attn in tqdm(attention_scores):
        # TODO: trim prefix tokens
        trim = topk * 2
        layer_attn = layer_attn[:, :, trim:, :]

        layer_result = layer_query_similarity(layer_attn, topk)
        layers_similarity.append(layer_result)

        # # TODO: first layer only
        # break

    layers_similarity = torch.stack(layers_similarity)
    # print(layers_similarity.shape)

    draw(layers_similarity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, required=True)
    args = parser.parse_args()
    main(args)
