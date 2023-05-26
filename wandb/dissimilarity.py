from alex_utils import *
import wandb
import os

import argparse
import torch
import numpy as np
import json
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 128


def tokenize_function(dataset):
    
    combined = np.column_stack((dataset["a_x"],dataset["a_y"])).tolist()
    
    tokenized_input_seq_pair = tokenizer(combined,
                                         max_length=max_length,
                                         return_token_type_ids=True, 
                                         padding='max_length',
                                         truncation=True)
    
    return tokenized_input_seq_pair


def main():
    global args
    global tokenizer
    
    arg_defaults = [
        ('--_tags', str, 'debug'), # NOTE: required if you use deploy_sweeps. please do not remove the _. Use 'debug' if you don't want wandb to sync.
        
        ('--seed', int, 42),
        ('--wdb_project', str, ''), # defaults to chdir, but will be overwritten if called as part of a sweep
        ('--wdb_entity', str, 'socialiq'),

        ('--fold', int, 0),
        ('--dataset_path', str, ''),
        ('--output_dir', str, ''),
        ('--cache_dir', str, ''),
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
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli', cache_dir = args.cache_dir)
    
    # set up 
    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    model = model.to(device)
    model.eval()
    
    #prepare data
    dataset = load_dataset("json", data_files={'train': os.path.join(args.dataset_path, 'siq_fold_' + str(args.fold) + '.jsonl')}, cache_dir=args.cache_dir)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    dataloader = DataLoader(tokenized_datasets["train"], shuffle=False, batch_size=16, collate_fn=lambda x: x)

    # run 
    all_entailments = [] 
    
    for batch in tqdm(dataloader):

        input_ids = torch.stack(tuple([torch.tensor(data["input_ids"]) for data in batch]))
        attention_mask = torch.stack(tuple([torch.tensor(data["attention_mask"]) for data in batch]))
        token_type_ids = torch.stack(tuple([torch.tensor(data["token_type_ids"]) for data in batch]))
        
        outputs = model(input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        token_type_ids=token_type_ids.to(device),
                        labels=None)

        predicted_probability = torch.softmax(outputs[0], dim=1)   # .cpu().detach()

        entailment = predicted_probability[:,0].tolist()

        all_entailments += entailment

        del input_ids
        del attention_mask
        del token_type_ids

        torch.cuda.empty_cache()
    
    # save results 
    orig_df = pd.read_json(path_or_buf= os.path.join(args.dataset_path, 'siq_fold_' + str(args.fold) + '.jsonl'), lines=True)
    orig_df['similarity'] = all_entailments
    
    if 'debug' not in args._tags:
        wandb.log({'train_loss': 0.1})
    
    os.makedirs(args.output_dir, exist_ok=True)
    orig_df.to_json(os.path.join(args.output_dir, 'siq_fold_' + str(args.fold) + '.jsonl'), orient='records', lines=True)

    

if __name__=='__main__':
    main()