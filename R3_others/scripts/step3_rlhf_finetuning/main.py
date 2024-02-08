#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss, critic_loss = trainer.train_rlhf(out)

"""

from audioop import avg
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed

from dschat.rlhf.ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from dschat.rlhf.rlhf_engine import DeepSpeedRLHFEngine
from dschat.utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer, \
    ExponentialMovingAverage
from dschat.utils.module.lora import convert_lora_to_linear_layer, convert_linear_layer_to_lora, make_model_gradient_checkpointing_compatible,only_optimize_lora_parameters
from dschat.utils.perf import print_throughput_step3
from deepspeed.accelerator import get_accelerator
from tqdm import tqdm
from nvitop import CudaDevice, ResourceMetricCollector
from nvitop.callbacks.tensorboard import add_scalar_dict



writer = None
import os
os.environ['NCCL_DEBUG'] = 'ERROR'

def parse_args():
    global writer
    parser = argparse.ArgumentParser(
        description="(Step 3) RLHF training arguments")

    parser.add_argument(
        '--test_data_type',
        type=str,
        default="gsm8k",
        help=""
    )
    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.'
    )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=1,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=1,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--generation_batches",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        type=int,
                        default=-1,
                        help="max steps to train")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bf16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')
    parser.add_argument(
        "--actor_dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the actor model."
    )
    parser.add_argument(
        "--critic_dropout",
        type=float,
        default=None,
        help="If critic dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the critic model."
    )
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial actor LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial critic LoRA learning rate (after the potential warmup period) to use."
    )
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')
    ## Mixed Precision ZeRO++
    parser.add_argument(
        '--enable_mixed_precision_lora',
        action='store_true',
        help='Enable Mixed Precision ZeRO++ for training and generation.')
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.'
        'This applies for both actor and critic models.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step3_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Actor/critic model overflow alignment
    parser.add_argument(
        '--align_overflow',
        action='store_true',
        help='Align loss scale overflow between actor and critic')
    ## Print actor model answers during training
    parser.add_argument('--print_answers',
                        action='store_true',
                        help='Print prompt and answers during training')
    parser.add_argument(
        "--print_answers_interval",
        type=int,
        default=1,
        help="If --print_answers enabled, controls the printing interval.")
    ## Testing
    parser.add_argument(
        '--enable_test_mode',
        action='store_true',
        help=
        'Enable a testing mode that terminates training based on args.test_stop_step'
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=0,
        help=
        "Training non-overflow step at which to terminate training during testing."
    )

    # do sampling
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--do_sample",
        type=int,
        default=0,
    )

    # save model by steps
    parser.add_argument(
        "--save_steps",
        type=int,
        default=0,
    )

    # check model rolling out
    parser.add_argument(
        "--response_start",
        type=str,
        default="### Response:",
    )

    # R3
    parser.add_argument(
        "--kl_ctl",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--clip_reward_value",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--cliprange",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--cliprange_value",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--reward_last_token",
        type=int,
        default=1,
    )


    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # args.tensorboard_path最好设置为最终的，不然可能这个参数在别的deepspeed有些地方用到，导致不一致

    if args.enable_tensorboard:
        # import tensorflow_io as tfio
        from datetime import datetime
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        exp_log_dir = f'{args.tensorboard_path}/step3_tensorboard_logs_{TIMESTAMP}'
        print(
            f"Tensorboard logs going to: {exp_log_dir}"
        )
        # writer = SummaryWriter(args.tensorboard_path)
        args.tensorboard_path= {args.tensorboard_path}

        cmd = f'hdfs dfs -mkdir -p {exp_log_dir}'
        # cmd = f''
        print(cmd)
        os.system(cmd)

        writer = SummaryWriter(exp_log_dir)

        collector = ResourceMetricCollector(devices=CudaDevice.all(),  # log all visible CUDA devices and use the CUDA ordinal
                                            root_pids={os.getpid()},   # only log the descendant processes of the current process
                                            interval=1.0)  



    # Validate settings
    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    if args.actor_zero_stage == 2 and args.critic_zero_stage == 2 and args.enable_hybrid_engine and args.offload and args.actor_lora_dim == 0:
        raise ValueError(
            "The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        )

    return args


import re


def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_prompt_seq_len,reload=True)

    print("length of dataset:")
    print(len(prompt_train_dataset))

    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_generation_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_generation_batch_size / args.per_device_training_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)
    print("length of dataset:")
    print(len(prompt_train_dataset))
    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.actor_model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    print("tokenizer len")
    print(len(tokenizer))

    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)


    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)

    # Mixed Precision ZeRO++
    if args.enable_mixed_precision_lora:
        assert args.actor_lora_dim > 0, "Mixed Precision LoRA requires LoRA to be enabled"
        assert args.actor_zero_stage == 3, "Mixed Precision LoRA requires Zero stage 3"
        rlhf_engine.actor.optimizer.quantize_nontrainable_params()
        print_rank_0("Mixed Precision ZeRO++ enabled")

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batches,
                                   args.per_device_training_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batches,
                                     args.per_device_training_batch_size)

    # Train!
    print_rank_0(
        f"***** Running training (total_iters={num_total_iters}) *****",
        args.global_rank)

    non_overflow_step_count = 0
    step_average_reward = 0.
    ema_reward_score = ExponentialMovingAverage()

    step_average_ans_len = 0.
    ema_ans_len = ExponentialMovingAverage()

    global_step = -1

    for epoch in range(args.num_train_epochs):
        epoch_start = time.time()
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank)
        if args.max_steps > 0 and global_step > args.max_steps:
            break

        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):

            global_step += 1

            # if global_step != 0 and args.save_steps > 0 and global_step % args.save_steps==0 and args.output_dir is not None:
            if args.save_steps > 0 and global_step % args.save_steps==0 and args.output_dir is not None:

                # save model by steps
                origin_output_dir = args.output_dir
                args.output_dir = args.output_dir+f"/step{global_step}"
                print_rank_0(f'saving model at step {global_step}...')
                rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
                rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
                if args.enable_ema:
                    rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                        rlhf_engine.actor_ema)
                

                if torch.distributed.get_rank() == 0:
                    save_hf_format(rlhf_engine.actor,
                                tokenizer,
                                args,
                                sub_folder='actor')
                    save_hf_format(rlhf_engine.critic,
                                tokenizer,
                                args,
                                sub_folder='critic')
                    if args.enable_ema:
                        save_hf_format(rlhf_engine.actor_ema,
                                    tokenizer,
                                    args,
                                    sub_folder='actor_ema')

                if args.actor_zero_stage == 3:
                    save_zero_three_model(rlhf_engine.actor,
                                        global_rank=args.global_rank,
                                        save_dir=os.path.join(
                                            args.output_dir, 'actor'),
                                        zero_stage=args.actor_zero_stage)
                    if args.enable_ema:
                        save_zero_three_model(rlhf_engine.actor_ema,
                                            global_rank=args.global_rank,
                                            save_dir=os.path.join(
                                                args.output_dir, 'actor_ema'),
                                            zero_stage=args.actor_zero_stage)
                if args.critic_zero_stage == 3:
                    save_zero_three_model(rlhf_engine.critic,
                                        global_rank=args.global_rank,
                                        save_dir=os.path.join(
                                            args.output_dir, 'critic'),
                                        zero_stage=args.critic_zero_stage)
                args.output_dir = origin_output_dir

                # LoRA 改回来？ TODO未测试
                if args.actor_lora_dim > 0:
                    rlhf_engine.actor = convert_linear_layer_to_lora(
                        rlhf_engine.actor, args.actor_lora_module_name,
                        args.actor_lora_dim)
                    if args.only_optimize_lora:
                        rlhf_engine.actor = only_optimize_lora_parameters(rlhf_engine.actor)
                        rlhf_engine.actor = make_model_gradient_checkpointing_compatible(
                            rlhf_engine.actor)
                if args.actor_lora_dim > 0:
                    rlhf_engine.actor_ema = convert_linear_layer_to_lora(
                        rlhf_engine.actor_ema, args.actor_lora_module_name,
                        args.actor_lora_dim)

                if args.critic_lora_dim > 0:
                    rlhf_engine.critic  = convert_linear_layer_to_lora(
                        rlhf_engine.critic , args.critic_lora_module_name,
                        args.critic_lora_dim)
                    if args.only_optimize_lora:
                        rlhf_engine.critic  = only_optimize_lora_parameters(rlhf_engine.critic )
                        rlhf_engine.critic  = make_model_gradient_checkpointing_compatible(
                            rlhf_engine.critic )

            if args.max_steps > 0 and global_step > args.max_steps:
                break

            batch_prompt = to_device(batch_prompt, device)

            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'],
                                              step,
                                              answer=batch_prompt["answer"],
                                              temperature=args.temperature,
                                              do_sample = args.do_sample,
                                              response_start=args.response_start
                                              )

            # ans_len = out["ans_len"].mean()
            # avg_ans_len = sum(ans_len) / len(ans_len)

            # avg_ans_len = get_all_reduce_mean(avg_ans_len).item()
            # ema_ans_len.update(avg_ans_len)
            # avg_ans_len = 0.

            training_start = time.time()
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_generation_batch_size])

            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                return_sum = 0
                advantage_sum = 0
                average_reward = 0
                average_ans_len = 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for ppo_ep in range(args.ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        # 8个进程，每个进程的exp_data大小都是1？所以他是八条一起更新？
                        actor_loss, critic_loss, advantage_avged_by_token_and_batch, return_avged_by_token_and_batch = trainer.train_rlhf(exp_data)
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        return_sum += return_avged_by_token_and_batch.item()
                        advantage_sum += advantage_avged_by_token_and_batch.item()


                        average_reward += exp_data["rewards"].mean()
                        average_ans_len += exp_data["ans_len"].mean()


                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsup_loss_sum += unsup_loss.item()

                        inner_iter += 1
                        if args.enable_ema:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)

                end = time.time()
                training_time = end - training_start
                e2e_time = training_time + trainer.generate_time * args.generation_batches  # it is an approximation, we did not include, e.g., rw forward time etc

                print_rank_0(
                    f'Epoch: {epoch} | Step: {step} | Total Steps: {len(prompt_train_dataloader)} | PPO Epoch: {ppo_ep+1} | Actor Loss: {actor_loss_sum/inner_iter} | Critic Loss: {critic_loss_sum/inner_iter} | Unsupervised Loss: {unsup_loss_sum/inner_iter} | Advantage: {advantage_sum/inner_iter} | Return: {return_sum/inner_iter}',
                    args.global_rank)
                print_throughput_step3(rlhf_engine.actor.module,
                                       rlhf_engine.critic, args, e2e_time,
                                       trainer.generate_time, training_time,
                                       args.global_rank)

                average_reward = get_all_reduce_mean(average_reward).item()
                step_average_reward += average_reward / args.gradient_accumulation_steps_actor
                if (step + 1) % args.gradient_accumulation_steps_actor == 0:
                    ema_reward_score.update(step_average_reward)
                    step_average_reward = 0.

                average_ans_len = get_all_reduce_mean(average_ans_len).item()
                step_average_ans_len += average_ans_len / args.gradient_accumulation_steps_actor
                if (step + 1) % args.gradient_accumulation_steps_actor == 0:
                    ema_ans_len.update(step_average_ans_len)
                    step_average_ans_len = 0.
                


                print_rank_0(
                    f"Average reward score: {average_reward/inner_iter} | EMA reward score: {ema_reward_score.get()}",
                    args.global_rank)
                print_rank_0(
                    f"Average answer len: {average_ans_len/inner_iter} | EMA answer len: {ema_ans_len.get()}",
                    args.global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)

                if args.enable_tensorboard and torch.distributed.get_rank(
                ) == 0:
                    print("Doing Tensorboard!")

                    writer.add_scalar('reward',
                                      average_reward / inner_iter,
                                      global_step=global_step)

                    writer.add_scalar('ans_len',
                                      average_ans_len / inner_iter,
                                      global_step=global_step)

                    if (step + 1) % args.gradient_accumulation_steps_actor == 0:
                        writer.add_scalar('ema_reward_score',
                                            ema_reward_score.get(),
                                        global_step=global_step)
                        writer.add_scalar('ema_answer_len',
                                        ema_ans_len.get(),
                                      global_step=global_step) 


                    # writer.add_scalar('ema_answer_len',
                    #                     ema_ans_len.get(),
                    #                   global_step=global_step)          

                    writer.add_scalar('actor_loss',
                                      actor_loss.item(),
                                      global_step=global_step)
                    writer.add_scalar('actor_loss_sum',
                                      actor_loss_sum,
                                      global_step=global_step)

                    writer.add_scalar('critic_loss',
                                      critic_loss.item(),
                                      global_step=global_step)
                    writer.add_scalar('critic_loss_sum',
                                      critic_loss_sum,
                                      global_step=global_step)

                    writer.add_scalar('advantage',
                                      advantage_avged_by_token_and_batch.item(),
                                      global_step=global_step)
                    writer.add_scalar('advantage_sum',
                                      advantage_sum,
                                      global_step=global_step)
                    writer.add_scalar('advantage_avg',
                                      advantage_sum/inner_iter,
                                      global_step=global_step)
                    writer.add_scalar('return',
                                      return_avged_by_token_and_batch.item(),
                                      global_step=global_step)
                    writer.add_scalar('return_sum',
                                      return_sum,
                                      global_step=global_step)
                    writer.add_scalar('return_avg',
                                      return_sum/inner_iter,
                                      global_step=global_step)
                    writer.flush()

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

            actor_overflow, critic_overflow = trainer.get_overflow()

            if not actor_overflow and not critic_overflow:
                non_overflow_step_count += 1

            if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                break

        if args.enable_test_mode:
            break
    
        epoch_end = time.time()
        print_rank_0(f'Epoch_time of epoch {epoch}: {epoch_end-epoch_start}')


        if args.output_dir is not None:
            origin_output_dir = args.output_dir
            args.output_dir = args.output_dir+f"/epoch{epoch}"
            print_rank_0('saving model ...')
            rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
            rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
            if args.enable_ema:
                rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                    rlhf_engine.actor_ema)
            

            if torch.distributed.get_rank() == 0:
                save_hf_format(rlhf_engine.actor,
                            tokenizer,
                            args,
                            sub_folder='actor')
                save_hf_format(rlhf_engine.critic,
                            tokenizer,
                            args,
                            sub_folder='critic')
                if args.enable_ema:
                    save_hf_format(rlhf_engine.actor_ema,
                                tokenizer,
                                args,
                                sub_folder='actor_ema')

            if args.actor_zero_stage == 3:
                save_zero_three_model(rlhf_engine.actor,
                                    global_rank=args.global_rank,
                                    save_dir=os.path.join(
                                        args.output_dir, 'actor'),
                                    zero_stage=args.actor_zero_stage)
                if args.enable_ema:
                    save_zero_three_model(rlhf_engine.actor_ema,
                                        global_rank=args.global_rank,
                                        save_dir=os.path.join(
                                            args.output_dir, 'actor_ema'),
                                        zero_stage=args.actor_zero_stage)
            if args.critic_zero_stage == 3:
                save_zero_three_model(rlhf_engine.critic,
                                    global_rank=args.global_rank,
                                    save_dir=os.path.join(
                                        args.output_dir, 'critic'),
                                    zero_stage=args.critic_zero_stage)
            args.output_dir = origin_output_dir

            # LoRA 改回来？ TODO未测试
            if args.actor_lora_dim > 0:
                rlhf_engine.actor = convert_linear_layer_to_lora(
                    rlhf_engine.actor, args.actor_lora_module_name,
                    args.actor_lora_dim)
                if args.only_optimize_lora:
                    rlhf_engine.actor = only_optimize_lora_parameters(rlhf_engine.actor)
                    rlhf_engine.actor = make_model_gradient_checkpointing_compatible(
                        rlhf_engine.actor)
            if args.actor_lora_dim > 0:
                rlhf_engine.actor_ema = convert_linear_layer_to_lora(
                    rlhf_engine.actor_ema, args.actor_lora_module_name,
                    args.actor_lora_dim)

            if args.critic_lora_dim > 0:
                rlhf_engine.critic  = convert_linear_layer_to_lora(
                    rlhf_engine.critic , args.critic_lora_module_name,
                    args.critic_lora_dim)
                if args.only_optimize_lora:
                    rlhf_engine.critic  = only_optimize_lora_parameters(rlhf_engine.critic )
                    rlhf_engine.critic  = make_model_gradient_checkpointing_compatible(
                        rlhf_engine.critic )


if __name__ == "__main__":
    main()
