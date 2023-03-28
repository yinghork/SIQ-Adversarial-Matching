import sys; sys.path.append('/work/yinghork/Alex_challenge/');
import logging
from alex_utils import *
import wandb

import argparse
import util
import torch
import numpy as np
import json

import pandas as pd
import random 
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment

from scipy.stats import zscore
import statistics
import collections
import math 
import os


def main():
    global args
    
    arg_defaults = [
        ('--_tags', str, 'debug'), # NOTE: required if you use deploy_sweeps. please do not remove the _. Use 'debug' if you don't want wandb to sync.
        
        ('--seed', int, 42),
        ('--wdb_project', str, ''), # defaults to chdir, but will be overwritten if called as part of a sweep
        ('--wdb_entity', str, 'socialiq'),

        ## TODO: add any other arguments you'd like
        ('--lam', float, 0.1),
        ('--fold', int, 0),
        ('--dataset_path_rel', str, ''),
        ('--dataset_path_sim', str, ''),
        ('--output_dir', str, ''),
    ]
    
    # TODO (optional): if there are any features of the argparse parser you want to add, initialize and pass in your parser here.
    parser = None
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_type", type=str, required=True, help="The model architecture",)
    args = process_defaults(arg_defaults, parser_in=parser)
    
    # TODO: copy set_seed() from alex_utils into this file if you'd like to use it (b/c of dependencies)
    # set_seed(args.seed)

    if 'debug' not in args._tags:
        wandb.init(
            project=args.wdb_project,
            entity=args.wdb_entity, 
            config=vars(args),
            tags=args._tags.split(','),
        )
        
    # import computed similarity + relevance values
    relevance_df = pd.read_json(path_or_buf= os.path.join(args.dataset_path_rel, 'siq_fold_' + str(args.fold) + '.jsonl'),lines=True)
    similarity_df = pd.read_json(path_or_buf= os.path.join(args.dataset_path_sim, 'siq_fold_' + str(args.fold) + '.jsonl'),lines=True)
    
        
    # create weight matrix 
    
    n = int(math.sqrt(len(relevance_df)))
    weight_matrix = np.zeros((n,n))
    lam = args.lam

    for ((i,relrow),(_,simrow)) in tqdm(zip(relevance_df.iterrows(), similarity_df.iterrows()), total=relevance_df.shape[0]):
    
        q_x = relrow["question_x"]
        q_y = relrow["question_y"]
        a_x = relrow["a_x"]
        a_y = relrow["a_y"]

        if(q_x != q_y and a_x != a_y):
            weight_matrix[i//n][i%n] = np.log(relrow["relevance"]) + lam * np.log(1 - simrow["similarity"])
        else:
            weight_matrix[i//n][i%n] = float("-inf")

    
    # do maximum bipartite matching
    row_ind, col_ind = linear_sum_assignment(weight_matrix, maximize = True)

    df_new = pd.DataFrame(columns=['video_id', 'qai_id', 'qid', 'ts', 'question', 'a', 'i', 'q_annotator', 'a_annotator', 'i_annotator'])
                          
    currlen = 0
    for row,col in zip(row_ind,col_ind):

        entry = relevance_df.iloc[row * n + col]

        video_id, qai_id, qid, ts = entry["video_id_x"], entry["qai_id_x"], entry["qid_x"], entry["ts_x"]
        question = entry["question_x"]
        a = entry["a_x"]
        i = entry["a_y"]

        df_new.loc[currlen] = [video_id, qai_id, qid, ts, question, a, i, "n/a", "n/a", "n/a"] 
        currlen += 1
    
    # save new matched data 
    os.makedirs(args.output_dir, exist_ok=True)
    df_new.to_json(os.path.join(args.output_dir, 'siq_fold_' + str(args.fold) + '_lambda_' + str(args.lam) + '.jsonl'), orient='records', lines=True)
                                
    
    if 'debug' not in args._tags:
        wandb.log({'train_loss': 0.1})

if __name__=='__main__':
    main()