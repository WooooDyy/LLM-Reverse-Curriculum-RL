# encoding: utf-8
import json
import argparse
import random
import time
import sys
sys.path.append("..")
sys.path.append("../..")
import logging
import sys
import re
from datasets import load_dataset
from datetime import datetime
from training.eval.llama2 import LlamaInterface
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model_name_or_path', type=str)
    args.add_argument('--data_path', type=str)
    args.add_argument('--results_path', type=str)
    args.add_argument('--mode', type=str, default="zero_shot")
    args.add_argument('--eval_size', type=int, default=-1)
    args.add_argument('--batch_size', type=int, default=4)

    args = args.parse_args()
    return args


def evaluate_batch(args):

    print(args.data_path)
    Testset = load_dataset("json",data_files=args.data_path)
    
    Testset = Testset["train"]
    if args.eval_size > 0:
        random_list = random.sample(range(len(Testset)), args.eval_size)
        print(random_list)
        Testset = Testset.select(random_list)
    print(Testset)


    peft_path = None
    llama = LlamaInterface(modelpath=args.model_name_or_path, peftpath=peft_path, add_lora=False)


    if args.mode == "few_shot":
        prompt_instruction = "Below is a question describing the task and several examples of correct solution to the task. Write a response that appropriately completes the request.\n\n"
        prompt_complex = open('../../prompts/prompt_original.txt').read()
    elif args.mode == "zero_shot":
        prompt_instruction = "Below is a question describing the task. Write a response that appropriately completes the request.\n\n"
        prompt_complex = ""

    cnt = 0
    with open( args.results_path, 'w') as f:

        batch_size = args.batch_size
        total_batches = (len(Testset) + batch_size - 1) // batch_size
        logger.info(f"Total Batch: {total_batches}")


        for i in tqdm(range(total_batches)):
    
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(Testset))
            batch = Testset.select(range(start_index, end_index))

            logger.info(f"Processing batch {i + 1}/{total_batches}:")

            prompts = []
            for item in batch:
                q = item['instruction']
                prompt_q = prompt_instruction + prompt_complex + '### Instruction:\n' + q + '\n\n' + '### Response:'
                prompts.append(prompt_q) 

            # for MNLI, SNLI, max_tokens=1024
            # for raceHigh, boardgame, max_tokens=2048
            responses = llama.generate_responses_from_llama(
                prompt=prompts,
                temperature=0,
                max_tokens=1024
            )

            logger.info("-------------------------------------")


            for idx, item in enumerate(batch):

                q = item['instruction']
                a = item['output']
                ans_ = responses[idx]
                print(f"Question: {item['instruction']}")
                print(f"Gold Answer: {item['output']}")
                print(f"Model Output: {ans_}")
                f.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))


if __name__ == '__main__':
    args = parse_args()
    print(args)

    evaluate_batch(args)
