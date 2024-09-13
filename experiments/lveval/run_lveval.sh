#!/bin/bash

MODEL_NAME="tinyllama-110M"
MAX_LEN=200000
MODEL=${MODELS_DIR}/${MODEL_NAME}
DATASET=${DATASETS_DIR}/LVEval

VNAME_EXT=""
SEG_SIZE=512
SEG_START=4096
BLOCK_SIZE=32
BUDGETS=2048
VNAME=ssize${SEG_SIZE}_sstart${SEG_START}_bsize${BLOCK_SIZE}_budget${BUDGETS}_${VNAME_EXT}
METHOD="-eattn"
OUTPUT_DIR=lvpred/$MODEL_NAME"_${VNAME}"$METHOD
python prediction.py --model-path $MODEL --model-name $MODEL_NAME --model-max-len $MAX_LEN --output-dir $OUTPUT_DIR --single-process --data-path $DATASET $METHOD \
  --segment_size $SEG_SIZE \
  --threshold_len $SEG_START \
  --block_size $BLOCK_SIZE \
  --budgets $BUDGETS \

VNAME_EXT=""
VNAME=base_${VNAME}
METHOD=""
OUTPUT_DIR=lvpred/$MODEL_NAME"_${VNAME}"$METHOD
python prediction.py --model-path $MODEL --model-name $MODEL_NAME --model-max-len $MAX_LEN --output-dir $OUTPUT_DIR --single-process --data-path $DATASET $METHOD





