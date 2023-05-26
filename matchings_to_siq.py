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


def shuffle_a_i(row):
    combined = [row["a"]] + row["i"]
    enum = list(enumerate(combined))
    random.shuffle(enum)
    
    indices, shuffled = zip(*enum)
    
    for i in range(len(combined)):
        val = "a" + str(i)
        row[val] = shuffled[i]
        
        if(indices[i] == 0):
            row["answer_idx"] = i
    
    return row 
    

def transform(df):
    df = df.drop(columns =['qai_id', 'q_annotator', 'a_annotator', 'i_annotator'])
                        
    df = df.rename(columns={"video_id": "vid_name", "question": "q"})  

    df = df.apply(lambda row: shuffle_a_i(row), axis=1)
    
    df = df.drop(columns =['a', 'i'])
    
    return df


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
    parser.add_argument("--lam", type=str, required=True, help="comma separated values of lam")
    parser.add_argument("--lam2", type=str, required=True, help="comma separated values of lam2")
    parser.add_argument("--lam3", type=str, required=True, help="comma separated values of lam3")
    args = parser.parse_args()
    return args

args = setup_args()
os.makedirs(args.output_dir, exist_ok=True)

for lam in args.lam.split(','):
    for lam2 in args.lam2.split(','):
        for lam3 in args.lam3.split(','):
        
            matched_df = pd.read_json(path_or_buf= os.path.join(args.dataset_path, 'siq_lam_' +str(lam) + '_lam2_' +str(lam2) + '_lam3_' +str(lam3) + '.jsonl'),lines=True)

            transformed_df = transform(matched_df)

            # Creating the siq dataset
            transformed_df.to_json(os.path.join(args.output_dir, 'siq_lam_' +str(lam) + '_lam2_' +str(lam2) + '_lam3_' +str(lam3) + '.jsonl'),orient='records',lines=True)

