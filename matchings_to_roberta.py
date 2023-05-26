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


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
    parser.add_argument("--type", type=str, required=True, help="train or valid")
    parser.add_argument("--lam", type=float, required=True, help="lambda that we want to generate")
    parser.add_argument("--lam2", type=float, required=True, help="lambda2 that we want to generate")
    parser.add_argument("--lam3", type=float, required=True, help="lambda3 that we want to generate")
    args = parser.parse_args()
    return args

args = setup_args()

matched_df = pd.read_json(path_or_buf= os.path.join(args.dataset_path, 'siq_lam_' +str(args.lam) + '_lam2_' +str(args.lam2) + '_lam3_' +str(args.lam3) + '.jsonl'),lines=True)

# Creating the roberta train dataset
os.makedirs(args.output_dir, exist_ok=True)
matched_df.to_json(os.path.join(args.output_dir, 'socialiq_permute_' + args.type + '.json'))
