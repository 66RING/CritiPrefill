#!/bin/bash

cd /xikexie/66ring/quick_prefill
export MODELS_DIR=/xikexie/66ring/models
export DATASETS_DIR=/xikexie/66ring/datasets
pip install transformers==4.39.2
pip install wandb
# pip install minference
# pip install triton==2.1.0

VNAME_EXT="block_q_max_min"

# export CUDA_VISIBLE_DEVICES=1,2
# MODEL_NAME="LWM-Text-Chat-1M"
# MODEL=${MODELS_DIR}/${MODEL_NAME}
# DATASET=${DATASETS_DIR}/LongBench
# MAX_LEN=1048576

# export SEG_SIZE=512
# export SEG_START=4096
# export BLOCK_SIZE=32
# export BUDGETS=2048
# export PREFILL_ONLY=1
# # VNAME=ssize${SEG_SIZE}_sstart${SEG_START}_bsize${BLOCK_SIZE}_budget${BUDGETS}_prefill${PREFILL_ONLY}_${VNAME_EXT}
# # METHOD="-eattn"
# VNAME=base
# METHOD=""
# python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --out_name base_"$MODEL_NAME"_${VNAME}"$METHOD" $METHOD

# # VNAME=ssize${SEG_SIZE}_sstart${SEG_START}_bsize${BLOCK_SIZE}_budget${BUDGETS}_prefill${PREFILL_ONLY}_${VNAME_EXT}
# # METHOD="-eattn"
# VNAME=minfer_${VNAME_EXT}
# METHOD=""
# python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --out_name base_"$MODEL_NAME"_${VNAME}"$METHOD" $METHOD


# exit

# =============================
# =============================
# =============================
# =============================


# Test Llama
#

# MODEL_NAME="Mistral-7B-Instruct-v0.2"
# MODEL=${MODELS_DIR}/${MODEL_NAME}
# DATASET=${DATASETS_DIR}/LongBench
# MAX_LEN=31500


# MODEL_NAME="tinyllama-110M"
# MODEL=${MODELS_DIR}/${MODEL_NAME}
# DATASET=${DATASETS_DIR}/LongBench
# MAX_LEN=1024


# test Llama-3
MODEL_NAME="Llama-3-8B-Instruct-Gradient-1048k"
MODEL=${MODELS_DIR}/${MODEL_NAME}
DATASET=${DATASETS_DIR}/LongBench
MAX_LEN=1048576
export SEG_SIZE=2048
export SEG_START=4096
export BLOCK_SIZE=32
export BUDGETS=512
export PREFILL_ONLY=1
VNAME=ssize${SEG_SIZE}_sstart${SEG_START}_bsize${BLOCK_SIZE}_budget${BUDGETS}_prefill${PREFILL_ONLY}_${VNAME_EXT}
METHOD="-eattn"
python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --out_name base_"$MODEL_NAME"_${VNAME}"$METHOD" $METHOD




# test LWM
MODEL_NAME="LWM-Text-Chat-1M"
MODEL=${MODELS_DIR}/${MODEL_NAME}
DATASET=${DATASETS_DIR}/LongBench
MAX_LEN=1048576
export SEG_SIZE=2048
export SEG_START=4096
export BLOCK_SIZE=32
export BUDGETS=512
export PREFILL_ONLY=1
VNAME=ssize${SEG_SIZE}_sstart${SEG_START}_bsize${BLOCK_SIZE}_budget${BUDGETS}_prefill${PREFILL_ONLY}_${VNAME_EXT}
METHOD="-eattn"
python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --out_name base_"$MODEL_NAME"_${VNAME}"$METHOD" $METHOD


# # base model
# METHOD=""
# python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --out_name base_"$MODEL_NAME"-"$METHOD" $METHOD



