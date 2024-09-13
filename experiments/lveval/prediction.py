import os
import re
import torch
import gc
import argparse
from tqdm import tqdm
from criti_prefill.modeling_patch import replace_llama_eattention, replace_base, criti_config

from config import (
    DATASET_MAXGEN, 
    DATASET_PROMPT, 
    DATASET_SELECTED, 
    DATASET_LENGTH_LEVEL,
)
from utils import (
    ensure_dir, 
    seed_everything,
    get_dataset_names,
    build_chat,
    post_process,
    model_generate,
    truncate_prompt,
    dump_preds_results,
    load_LVEval_dataset,
    load_model_and_tokenizer_once,
)


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    model_name,
):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        prompt = truncate_prompt(tokenizer, prompt, max_length)

        if hasattr(model, "chat"): # using model.chat() to generate prediction, which determines the conversation format of the open-source model by itself.
            pred = model.chat(
                tokenizer, 
                prompt, 
                max_new_tokens=max_gen,
                max_length=None,
                do_sample=False,
                history=[], #None,
            )[0]
        else: # using model.generate() to generate prediction, make sure custmized conversation format in build_chat() is correct.
            prompt = build_chat(tokenizer, prompt, model_name)
            pred = model_generate(tokenizer, prompt, max_gen, model)

        pred, prefill_time = model_generate(tokenizer, prompt, max_gen, model)

        pred = post_process(pred, model_name)

        print(f"pred: {pred}", flush=True)
        preds.append({
            "pred": pred,
            "prefill_time": prefill_time,
            "answers": json_obj["answers"],
            "gold_ans": json_obj["answer_keywords"] if "answer_keywords" in json_obj else None,
            "input": json_obj["input"],
            "all_classes": json_obj["all_classes"] if "all_classes" in json_obj else None,
            "length": json_obj["length"],
        })
    return preds

def evaluate(mix):
    datas = mix["datas"]
    dataset = mix["dataset"]
    id = mix["id"]
    dataset_name = re.split('_.{1,3}k', dataset)[0]
    prompt_format = mix["dataset_prompt"][dataset_name]
    max_gen = mix["dataset_maxgen"][dataset_name]
    preds = get_pred(
        mix["model"],
        mix["tokenizer"],
        datas,
        mix["max_length"],
        max_gen,
        prompt_format,
        mix["model_name"],
    )
    mix["shared_dict"][id] = preds

def split_datasets(input_list, num_parts, dataset, shared_dict, device_dict, args):
    avg = len(input_list) // num_parts
    remainder = len(input_list) % num_parts
    result = []
    start = 0
    id = 0
    for i in range(num_parts):
        end = start + avg
        if i < remainder:
            end += 1
        dic = {}
        dic["datas"] = input_list[start:end]
        dic["dataset"] = dataset
        dic["id"] = id
        dic["model_path"] = args.model_path
        dic["model_name"] = args.model_name
        dic["max_length"] = args.model_max_length
        dic["data_path"] = args.data_path
        dic["out_dir"] = args.output_dir
        dic["model"] = device_dict[id][0]
        dic["tokenizer"] = device_dict[id][1]
        dic["shared_dict"] = shared_dict
        dic["dataset_prompt"] = DATASET_PROMPT
        dic["dataset_maxgen"] = DATASET_MAXGEN
        dic["args"] = args
        result.append(dic)
        start = end
        id = id + 1
    return result

def multiple_processing_once(num_gpus, dataset, shared_dict, device_dict, args):
    datas = load_LVEval_dataset(dataset, args.data_path)
    mixs = split_datasets(datas, num_gpus, dataset, shared_dict, device_dict, args)
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=num_gpus)
    pool.map(evaluate, mixs)
    pool.close()
    pool.join()
    merged_results = []
    for i in range(len(shared_dict)):
        merged_results += shared_dict[i]
    dump_preds_results(merged_results, os.path.join(args.output_dir, dataset + ".jsonl"))

def load_model_and_tokenizer_serial(num_gpus, model_path):
    device_dict = dict()
    for i in range(num_gpus):
        load_model_and_tokenizer_once(i, model_path, device_dict)
    return device_dict

def multiple_processing(datasets, args):
    num_gpus = torch.cuda.device_count()
    device_dict = load_model_and_tokenizer_serial(num_gpus, args.model_path)
    manager = torch.multiprocessing.Manager()
    shared_dict = manager.dict()
    for dataset in tqdm(datasets):
        multiple_processing_once(num_gpus, dataset, shared_dict, device_dict, args)

def single_processing(datasets, args):
    if "glm" in args.model_path.lower():
        print("loading glm")
        from modeling_chatglm import ChatGLMForConditionalGeneration
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        model = ChatGLMForConditionalGeneration.from_pretrained(
            args.model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        # TODO: remove
    else:
        model, tokenizer = load_model_and_tokenizer_once(-1, args.model_path)

    criti_config(model,
                 segment_size=args.segment_size,
                 threshold_len=args.threshold_len,
                 block_size=args.block_size,
                 budgets=args.budgets,
                 layer_fusion=args.layer_fusion,
                 layer_skip=args.layer_skip)

    if "llama-3" in args.model_path.lower():
        tokenizer.add_special_tokens({'eos_token': '<|eot_id|>'})

    for dataset in tqdm(datasets):
        datas = load_LVEval_dataset(dataset, args.data_path)

        dataset_name = re.split('_.{1,3}k', dataset)[0]
        prompt_format = DATASET_PROMPT[dataset_name]
        max_gen = DATASET_MAXGEN[dataset_name]
        preds = get_pred(
            model,
            tokenizer,
            datas,
            args.model_max_length,
            max_gen,
            prompt_format,
            args.model_name,
        )
        dump_preds_results(preds, os.path.join(args.output_dir, dataset + ".jsonl"))
        gc.collect()
        torch.cuda.empty_cache()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-max-length", type=int, default=15500)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--single-process", action="store_true")

    parser.add_argument("-eattn", "--enable_eattention", action='store_true')

    parser.add_argument("--segment_size", type=int, default=512)
    parser.add_argument("--threshold_len", type=int, default=1024)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--budgets", type=int, default=2048)
    parser.add_argument("--layer_skip", type=int, default=1)
    parser.add_argument("--layer_fusion", action='store_true')
    return process_args(parser.parse_args(args))

def process_args(args):
    model_path = args.model_path.rstrip("/")
    if not args.model_name:
        args.model_name = os.path.basename(model_path)
    return args

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    if args.enable_eattention:
        replace_llama_eattention()
    else:
        replace_base()

    ensure_dir(args.output_dir)
    datasets = get_dataset_names(DATASET_SELECTED, DATASET_LENGTH_LEVEL)

    if args.single_process:
        single_processing(datasets, args)
    else:
        multiple_processing(datasets, args)
