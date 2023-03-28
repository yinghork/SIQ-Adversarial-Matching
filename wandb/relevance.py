import sys; sys.path.append('/work/yinghork/Alex_challenge/');
import logging
from alex_utils import *
import wandb
import os

import argparse
import util
import torch
import numpy as np
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

import pandas as pd

from transformers import RobertaTokenizer
from train_roberta_multiple_choice import RobertaMultipleChoiceTask

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

logging.disable(logging.WARNING)

def setup_args(wandb_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset name: movieqa, tvqa, socialiq_a5, socialiq_a2, socialiq_a4")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
    parser.add_argument("--resume_from", type=str, default=None, help="resume from this checkpoint file")
    parser.add_argument("--learning_rate", type=float, required=True, help="learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--freeze_lm", default=False, action='store_true',
                        help="whether to freeze language model parameters")
    parser.add_argument("--load_pretrained_lm", default=False, action='store_true',
                        help="whether to load pretrained language model parameters")
    parser.add_argument("--half_precision", default=False, action='store_true', help="whether to use half precision")
    parser.add_argument("--inference_only", default=False, action='store_true',
                        help="set this if we are not training but just evaluating")
    parser.add_argument("--do_test", default=False, action='store_true',
                        help="set this if we are evaluating on test set and want to output prediction into test_output.txt")
    parser.add_argument("--output_prediction_correctness", default=False, action='store_true',
                        help="set this if we want to output prediction correctness into prediction_correctness.json")
    parser.add_argument("--valid_subset_frac", type=float, default=1., help="fraction of validation set to use")
    parser.add_argument("--max_epochs", type=int, default=20, help="max epochs to train")
    parser.add_argument("--do_name_that_annotator", default=False, action='store_true', help="run the script for the Name That Annotator! task")
    parser.add_argument("--annotator_map_dict", type=str, required=('--do_name_that_annotator' in sys.argv), help="the path to the dict of annotator mapping, should be a json file")
    parser.add_argument("--do_answer_only", default=False, action='store_true',
                        help="set this if we are doint the answer only task")
    
    model_args = parser.parse_args(['--dataset_path=/work/yinghork/data/old_socialiq_data',
                              '--dataset_name=socialiq_permute_a2',
                              '--output_dir=/work/yinghork/results/socialiq/saliency_map_debug',
                              '--learning_rate=1e-6',
                              '--batch_size=3',
                              '--resume_from=' + wandb_args.model_path
                             ])
    
    return model_args

def prepare_features(question, answer, max_qa_seq_length=40, padding=True):
    # Tokenizne Input
    indexed_tokens = tokenizer.encode(question, answer, max_length=max_qa_seq_length, add_special_tokens=True)

    # Input Mask
    input_mask = [1] * len(indexed_tokens)
    # Pad to max_qa_seq_length using padding special token
    if padding:
        while len(indexed_tokens) < max_qa_seq_length:
            indexed_tokens.append(tokenizer.pad_token_id)
            input_mask.append(0)
    
    return torch.tensor(indexed_tokens).unsqueeze(dim=0), input_mask


class SocialIQ_QA_Relevance_Classification(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = self.data.question_x[index]
        a = self.data.a_y[index]
        X_a, _ = prepare_features(question, a)
        return index, X_a

    def __len__(self):
        return self.len


def main():
    global args
    
    arg_defaults = [
        ('--_tags', str, 'debug'), # NOTE: required if you use deploy_sweeps. please do not remove the _. Use 'debug' if you don't want wandb to sync.
        
        ('--seed', int, 42),
        ('--wdb_project', str, ''), # defaults to chdir, but will be overwritten if called as part of a sweep
        ('--wdb_entity', str, 'socialiq'),

        ## TODO: add any other arguments you'd like
        ('--fold', int, 0),
        ('--model_path', str, ''),
        ('--dataset_path', str, ''),
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
    
    # set up 
    model_args = setup_args(args)
    util.set_seed(2020)

    task = RobertaMultipleChoiceTask(model_args)
    model = task.model.eval()
    
    
    # prepare data
    batch_size = 32

    params = {'batch_size': batch_size,
              'shuffle': False,
              'drop_last': False,
              }

    df = pd.read_json(path_or_buf= os.path.join(args.dataset_path, 'siq_fold_' + str(args.fold) + '.jsonl'),lines=True)
    dataset = SocialIQ_QA_Relevance_Classification(df)
    
    loader = DataLoader(dataset, num_workers=0, **params)
    

    # run 
    logits = np.zeros(len(dataset))
    total = 0

    for i,value_to_unpack in enumerate(tqdm(loader)):

        index, X_a = value_to_unpack

        X_a = X_a.squeeze(1)
        X_a = X_a.to(task.device)
        logit_a, embedding_output_a = model(X_a, return_embedding_output=True)
        logit_a = logit_a[0].cpu().detach().numpy()

        # use sigmoid to transform logit to probability 
        logit_a = 1/(1 + np.exp(-logit_a))

        logit_a = logit_a.squeeze()

        logits[total:total+len(logit_a)] += logit_a

        total += len(logit_a)
    
    # save results 
    orig_df = pd.read_json(path_or_buf= os.path.join(args.dataset_path, 'siq_fold_' + str(args.fold) + '.jsonl'),lines=True)
    orig_df['relevance'] = logits
    
    os.makedirs(args.output_dir, exist_ok=True)
    orig_df.to_json(os.path.join(args.output_dir, 'siq_fold_' + str(args.fold) + '.jsonl'), orient='records', lines=True)
    
    if 'debug' not in args._tags:
        wandb.log({'train_loss': 0.1})


if __name__=='__main__':
    main()