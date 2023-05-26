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
    parser.add_argument("--lam", type=str, required=True, help="comma separated values of lam")
    parser.add_argument("--lam2", type=str, required=True, help="comma separated values of lam2")
    parser.add_argument("--lam3", type=str, required=True, help="comma separated values of lam3")
    args = parser.parse_args()
    return args

args = setup_args()
os.makedirs(args.output_dir, exist_ok=True)

# import computed similarity + relevance values
dfs = []

for lam in args.lam.split(','):
    for lam2 in args.lam2.split(','):
        for lam3 in args.lam3.split(','):
            dfs = []

            for fold in range(args.num_folds):
                matching = pd.read_json(path_or_buf= os.path.join(args.dataset_path, 'lam_' + lam, 'lam2_' + lam2, 'lam3_' + lam3, 'siq_fold_' + str(fold) + '.jsonl'),lines=True)
                dfs.append(matching)

            combined_dfs = pd.concat(dfs)
            combined_dfs.to_json(os.path.join(args.output_dir, 'siq_lam_' + lam + '_lam2_' + lam2 + '_lam3_' + lam3 + '.jsonl'), orient='records', lines=True)

    