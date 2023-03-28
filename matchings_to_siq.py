import os
import pandas as pd
import random 
import numpy as np
import torch
from tqdm import tqdm
import argparse

from scipy.stats import zscore
import statistics
import math 
import collections
import random


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
    parser.add_argument("--type", type=str, required=True, help="train or valid")
    parser.add_argument("--lam", type=float, required=True, help="lambda that we want to generate")
    args = parser.parse_args()
    return args

args = setup_args()

matched_df = pd.read_json(path_or_buf= os.path.join(args.dataset_path, 'siq_lambda_' +str(args.lam) + '.jsonl'),lines=True)

matched_df = matched_df.drop(columns =['qai_id', 'q_annotator', 'a_annotator', 'i_annotator'])
                        
matched_df = matched_df.rename(columns={"video_id": "vid_name", "question": "q"})  

def shuffle_a_i(row):
    if(random.uniform(0, 1) >= .5):
        row["a0"],row["a1"],row["answer_idx"] = row["a"], row["i"], 0
    else:
        row["a0"],row["a1"],row["answer_idx"] = row["i"], row["a"], 1
    return row

matched_df = matched_df.apply(lambda row: shuffle_a_i(row), axis=1)
matched_df = matched_df.drop(columns =['a', 'i'])

# Creating the roberta train dataset
os.makedirs(args.output_dir, exist_ok=True)
matched_df.to_json(os.path.join(args.output_dir, 'matched_siq_' + args.type + '.jsonl'),orient='records',lines=True)

