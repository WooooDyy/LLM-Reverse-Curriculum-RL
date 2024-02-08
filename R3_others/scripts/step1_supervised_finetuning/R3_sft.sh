#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# SFT script
# zero_stage can be 2 or 3
# for math datasets, train_epochs=2
# for other datasets, train_epochs=5
ZERO_STAGE=2
learning_rate=2e-5
num_train_epochs=3

model_name_or_path="/your_path_to_llama-2-7b-hf"
data_path="../../data/gsm8k_cot/gsm8k_nl_train_example.json"
output_base="/your_output_model_dir"
output_dir=${output_base}lr${learning_rate}_ep${num_train_epochs}/
data_output_path="/your_output_data_dir_to_save_shuffle_index"

mkdir -p $output_dir

# for GSM8K-P-CoT, max_seq_len=1024
# for other math datasets, max_seq_len=512 
# for MNLI, SNLI, max_seq_len=512
# for raceHigh, raceMiddle, max_seq_len=1024
# for boardgame, max_seq_len=512
# and we keep total_batch_size=256
deepspeed \
    --master_port 39000 \
    --num_gpus 8 \
    main.py \
    --model_name_or_path $model_name_or_path\
    --data_path $data_path \
    --data_split "10,0,0" \
    --data_output_path $data_output_path \
    --learning_rate $learning_rate \
    --zero_stage $ZERO_STAGE \
    --gradient_accumulation_steps 8  \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --per_device_eval_batch_size 4 \
    --max_seq_len 512 \
    --print_loss  \
    --num_train_epochs ${num_train_epochs} \
    --deepspeed \
    --output_dir  ${output_dir} \
    > step1_llama_base_7b_gsm8k_cot.log 2>&1
