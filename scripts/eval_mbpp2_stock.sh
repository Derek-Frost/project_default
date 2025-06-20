#!/bin/bash

# Task Selection
TASK="mbpp2" # Available options: mbpp2, math, ai2_arc

# First Stage Inference: Classification Expert
# Set to 'None' if not using cls expert
CLS_EXPERT_PATH="None"

# Second Stage: Expert Models
# Set your actual model paths
CODE_EXPERT_PATH=""
MATH_EXPERT_PATH=""
REASONING_EXPERT_PATH=""

# Start evaluation!
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \
    base_model@_global_=llama3i8b \
    task@_global_=$TASK \
    mode@_global_=eval \
    prompt_based_eval=True \
    experts_path_dict.code=$CODE_EXPERT_PATH \
    experts_path_dict.math=$MATH_EXPERT_PATH \
    experts_path_dict.reasoning=$REASONING_EXPERT_PATH \
    load_ckpt=$CLS_EXPERT_PATH
