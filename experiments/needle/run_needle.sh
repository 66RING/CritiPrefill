#!/bin/bash

START=4096
END=140000
STEP=4096

# TODO: review layer fusion default value

MODEL_NAMES=(
  # "Llama-3-8B-Instruct-Gradient-1048k"
  # "Yi-9B-200K"
  "tinyllama-110M"
)

VNAME_EXT=""

for MODEL_NAME in "${MODEL_NAMES[@]}"; do

SEG_SIZE=512
SEG_START=4096
BLOCK_SIZE=32
BUDGETS=1024
VNAME=ssize${SEG_SIZE}_sstart${SEG_START}_bsize${BLOCK_SIZE}_budget${BUDGETS}_layer_fusion_on_${VNAME_EXT}
METHOD="-eattn"
  python -u run_needle_in_haystack.py --s_len $START --e_len $END \
      --model_name $MODELS_DIR/${MODEL_NAME} \
      --attn_implementation flash_attention_2 \
      --step $STEP \
      --model_version ${MODEL_NAME}_${START}_${END}_${STEP}_${VNAME}${METHOD}\
      --segment_size $SEG_SIZE \
      --threshold_len $SEG_START \
      --block_size $BLOCK_SIZE \
      --budgets $BUDGETS \
      --layer_fusion \
      $METHOD

VNAME=base_${VNAME_EXT}
METHOD=""
  # TODO: HUG there
  python -u run_needle_in_haystack.py --s_len $START --e_len $END \
      --model_name $MODELS_DIR/${MODEL_NAME} \
      --attn_implementation flash_attention_2 \
      --step $STEP \
      --model_version ${MODEL_NAME}_${START}_${END}_${STEP}_${VNAME}${METHOD}\
      $METHOD
done


