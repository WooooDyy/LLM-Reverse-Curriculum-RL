import json
import argparse
import random
import time
import logging
import sys
import re
from datasets import load_dataset
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# extracting answers, you can modify the answer format as you want
# MNLI example
def test_answer(pred_str, ans_str):
    pattern = "###"
    
    pred = pred_str.split(pattern)
    gold = ans_str.split(pattern)
    
    if(len(pred) > 1):
        pred_answer = pred[1].strip().lower().replace('.','')
        gold_answer = gold[1].strip().lower().replace('.','')
        # print(gold_answer)
        return pred_answer == gold_answer

    else: 
        return False

def parse_pred_ans(filename):
    with open(filename) as fd: lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = 'none'
    questions = []
    ans_pred = []
    ans_gold = []
    right = []
    for l in lines:
        if(l.startswith('Q: ')):
            if(am is not None and a is not None):
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if(test_answer(am, a)):
                    right.append(1)
                    acc += 1
                else:
                    right.append(0)
            current_mode = 'q'
            q = l
            num_q += 1
        elif(l.startswith('A_model:')):
            current_mode = 'am'
            am = l
        elif(l.startswith('A:')):
            current_mode = 'a'
            a = l
        else:
            if(current_mode == 'q'): q += l
            elif(current_mode == 'am'): am += l
            elif(current_mode == 'a'): a += l
            else:
                raise ValueError(current_mode)
                
    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    if(test_answer(am, a)):
        right.append(1)
        acc += 1
    else:
        right.append(0)
    print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold, right


if __name__ =="__main__":
    # parse_pred_ans("eval_mnli/your_test_file.txt")
    parse_pred_ans("eval_mnli/R3_test.txt")


            