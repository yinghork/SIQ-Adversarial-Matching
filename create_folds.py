# Run this file to generate train/valid files that the RoBERTa model can read 

import argparse
import pandas as pd
import os
import json
import numpy as np

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
    parser.add_argument("--num_folds", type=int, required=True, help="number of folds for data")
    args = parser.parse_args()
    return args

def generate_folds(df):
    # get question, answer
    df = df.drop_duplicates(subset = ['question', 'a'],keep = 'first')
    df = df.reset_index(drop=True)

    # split into folds
    shuffled = df.sample(frac=1)
    result = np.array_split(shuffled, args.num_folds)

    # write output

    for i,fold in enumerate(result):
        result[i] = fold.reset_index().rename(columns={"index":"orig_index"})
        result[i]['key'] = 1
        result[i] = pd.merge(result[i],result[i], on ='key').drop("key", 1)
        result[i].to_json(os.path.join(args.output_dir, 'siq_fold_' + str(i) + '.jsonl'), orient='records', lines=True)

        
args = setup_args()
os.makedirs(args.output_dir, exist_ok=True)

# get original SIQ data 

df = pd.read_json(args.dataset_path)
generate_folds(df)

    