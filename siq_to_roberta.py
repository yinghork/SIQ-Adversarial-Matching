# Run this file to generate train/valid files that the RoBERTa model can read 

import argparse
import pandas as pd
import os
import json

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
    args = parser.parse_args()
    return args

def clean_data(df):
    df_a0 = df.loc[df['answer_idx'] == 0]
    df_a0 = df_a0.rename(columns={"a0": "a", "a1": "i"})
    df_a1 = df.loc[df['answer_idx'] == 1]
    df_a1 = df_a1.rename(columns={"a1": "a", "a0": "i"})

    df_clean = pd.concat([df_a0, df_a1], sort=True)
    df_clean = df_clean.rename(columns={"q" : "question", "vid_name" : "video_id"})
    df_clean["q_annotator"] = ['n/a' for i in range(len(df_clean))]
    df_clean["a_annotator"] = ['n/a' for i in range(len(df_clean))]
    df_clean["i_annotator"] = ['n/a' for i in range(len(df_clean))]
    df_clean["qai_id"] = [i for i in range(len(df_clean))]
    df_clean = df_clean.drop(columns=['answer_idx'])
    
    return df_clean

args = setup_args()

# get original SIQ data 

dataset_path = args.dataset_path
df_train = pd.read_json(os.path.join(dataset_path, 'siq_train2.jsonl'),lines=True)
df_valid = pd.read_json(os.path.join(dataset_path, 'siq_val2.jsonl'),lines=True)

# clean train data
df_train_clean = clean_data(df_train)

# clean val data
df_valid_clean = clean_data(df_valid)

os.makedirs(args.output_dir, exist_ok=True)
df_train_clean.to_json(os.path.join(args.output_dir, 'socialiq_permute_train.json'))
df_valid_clean.to_json(os.path.join(args.output_dir, 'socialiq_permute_valid.json'))