import os
import pandas as pd
import random 
import numpy as np
import torch
from tqdm import tqdm
import argparse

from scipy.optimize import linear_sum_assignment

from scipy.stats import zscore
import statistics
import math 
import collections


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
    parser.add_argument("--num_folds", type=int, required=True, help="number of folds to combine")
    args = parser.parse_args()
    return args

args = setup_args()
os.makedirs(args.output_dir, exist_ok=True)

# import computed similarity + relevance values
dfs = []

for lam in [0.0,0.01,0.1,0.5,1.0]:
    dfs = []
    
    for fold in range(args.num_folds):
        matching = pd.read_json(path_or_buf= os.path.join(args.dataset_path, 'siq_fold_' + str(fold) + '_lambda_' + str(lam) + '.jsonl'),lines=True)
        dfs.append(matching)
        
    combined_dfs = pd.concat(dfs)
    combined_dfs.to_json(os.path.join(args.output_dir, 'siq_lambda_' + str(lam) + '.jsonl'), orient='records', lines=True)

    