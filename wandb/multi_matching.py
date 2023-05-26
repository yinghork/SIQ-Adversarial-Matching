import logging
from alex_utils import *
import wandb

import argparse
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


def compute_matching_value(q_x, q_y, a_x, a_y, rel, sim, dissim):
    
    if(q_x != q_y and a_x != a_y):
        return args.lam * np.log(rel) + args.lam2 * np.log(1 - dissim) + args.lam3 * np.log(sim)
    else:
        return float("-inf")
    

def main():
    global args
    
    arg_defaults = [
        ('--_tags', str, 'debug'), # NOTE: required if you use deploy_sweeps. please do not remove the _. Use 'debug' if you don't want wandb to sync.
        
        ('--seed', int, 42),
        ('--wdb_project', str, ''), # defaults to chdir, but will be overwritten if called as part of a sweep
        ('--wdb_entity', str, 'socialiq'),

        ## TODO: add any other arguments you'd like
        ('--lam', float, 0.1),
        ('--lam2', float, 0.1),
        ('--lam3', float, 0.1),
        ('--fold', int, 0),
        ('--num_matchings', int, 5),
        ('--dataset_path_rel', str, ''),
        ('--dataset_path_sim', str, ''),
        ('--dataset_path_dissim', str, ''),
        ('--output_multi_matching_dir', str, ''),
        ('--all_values_dir', str, ''),
    ]
    

    parser = None
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
    dissimilarity_df = pd.read_json(path_or_buf= os.path.join(args.dataset_path_dissim, 'siq_fold_' + str(args.fold) + '.jsonl'),lines=True)
    
        
    # create weight matrix 
    
    n = int(math.sqrt(len(relevance_df)))
    
    combined_df = relevance_df.copy()
    combined_df["similarity"] = similarity_df["similarity"]
    combined_df["dissimilarity"] = dissimilarity_df["similarity"]
    
    matching_df = combined_df.apply(lambda x: compute_matching_value(x.question_x, x.question_y, x.a_x, x.a_y, x.relevance, x.similarity, x.dissimilarity), axis=1)
    
    combined_df["matching"] = matching_df
    
    weight_matrix = matching_df.to_numpy().reshape((n, n))
    
    # save combined_df (one dataframe with relevance + similarity + matching values)
    os.makedirs(os.path.join(args.all_values_dir, 'lam_' + str(args.lam), 'lam2_' + str(args.lam2), 'lam3_' + str(args.lam3)), exist_ok=True)
    combined_df.to_json(os.path.join(args.all_values_dir, 'lam_' + str(args.lam), 'lam2_' + str(args.lam2), 'lam3_' + str(args.lam3), 'siq_fold_' + str(args.fold) + '.jsonl'), orient='records', lines=True)

    
    # do multiple maximum bipartite matching
    
    row_ind, col_ind = linear_sum_assignment(weight_matrix, maximize = True)

    df_new = pd.DataFrame(columns=['video_id', 'qai_id', 'qid', 'ts', 'question', 'a', 'i', 'q_annotator', 'a_annotator', 'i_annotator'])
                          
    for row,col in zip(row_ind,col_ind):

        entry = combined_df.iloc[row * n + col]

        video_id, qai_id, qid, ts = entry["video_id_x"], entry["qai_id_x"], entry["qid_x"], entry["ts_x"]
        question = entry["question_x"]
        a = entry["a_x"]
        i = entry["a_y"]

        df_new.loc[row] = [video_id, qai_id, qid, ts, question, a, [i], "n/a", "n/a", "n/a"] 
    
    
    # iterate for more matchings 
    for i in range(args.num_matchings):
        
        # take out current matchings
        for row,col in zip(row_ind,col_ind):
            weight_matrix[row][col] = float("-inf")
            matched_ans = combined_df.loc[row * n + col, "a_y"]
            for othercol in range(n):
                if(combined_df.loc[row * n + othercol, "a_y"] == matched_ans):
                    weight_matrix[row][othercol] = float("-inf")
            
        row_ind, col_ind = linear_sum_assignment(weight_matrix, maximize = True)

        for row,col in zip(row_ind,col_ind):
            entry = combined_df.iloc[row * n + col]
            i = entry["a_y"]
            df_new.loc[row,"i"].append(i)
            
    
    # save new matched data 
    os.makedirs(os.path.join(args.output_multi_matching_dir, 'lam_' + str(args.lam), 'lam2_' + str(args.lam2), 'lam3_' + str(args.lam3)), exist_ok=True)
    df_new.to_json(os.path.join(args.output_multi_matching_dir, 'lam_' + str(args.lam), 'lam2_' + str(args.lam2), 'lam3_' + str(args.lam3), 'siq_fold_' + str(args.fold) + '.jsonl'), orient='records', lines=True)
                                
    
    if 'debug' not in args._tags:
        wandb.log({'train_loss': 0.1})

if __name__=='__main__':
    main()