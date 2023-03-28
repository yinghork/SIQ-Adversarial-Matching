# SIQ-Adversarial-Matching


### 1. How to train relevance model (given the base SIQ data): 
- Your train + validation file must be named siq_train2.jsonl and siq_val2.jsonl, respectively
- Run the following command to get the train + validation file into the RoBERTa format that the model reads:

```
python siq_to_roberta.py --dataset_path /work/yinghork/socialiq_data --output_dir /work/yinghork/socialiq_data
```

- Run the following command with updated dataset paths and output directory: 
    
```
python train_roberta_multiple_choice.py --dataset_path /work/yinghork/socialiq_data --dataset_name socialiq_permute_a2 --output_dir /work/yinghork/socialiq_data/matching_lr_1e-6_bs_3_unfreeze --learning_rate 1e-6 --batch_size 3
```

- Use the best model’s weights in the next steps


### 2. How to get adversarial matchings given some initial matchings and relevance model:
- Create folds from the Roberta dataset created in step 1: 

```
python create_folds.py --num_folds 16 --dataset_path /work/yinghork/socialiq_data/socialiq_permute_train.json --output_dir /work/yinghork/socialiq_data/folds/train
python create_folds.py --num_folds 2 --dataset_path /work/yinghork/socialiq_data/socialiq_permute_valid.json --output_dir /work/yinghork/socialiq_data/folds/valid
```

- For each of the train and validation data: 
    - Run the relevance model sweep:
        - Edit relevance_grid.yml file given the "TODO"
        - Run: ```wandb sweep relevance_grid.yml```
        - Run sbatch with the wandb agent command
    - Run the similarity model sweep:
        - Edit similarity_grid.yml file given the "TODO"
        - Run: ```wandb sweep similarity_grid.yml```
        - Run sbatch with the wandb agent command
    - Run the matching sweep: 
        - Edit matching_grid.yml file
        - Run: ```wandb sweep matching_grid.yml```
        - Run sbatch with the wandb agent command
    - Combine the folds: 
        - ```python combine_folds.py --num_folds 1 --dataset_path /work/yinghork/socialiq_data/matchings/train --output_dir /work/yinghork/socialiq_data/matchings/train```

### 3. How to retrain relevance model on new matchings: 
- For each of the train and validation data: 
    - Get the matching data into the roberta format that the model reads (two files: 1 train file and 1 validation file, and both have to be in the same folder):
```
python matchings_to_roberta.py --type train --lam 0.1 --dataset_path /work/yinghork/socialiq_data/matchings/train --output_dir /work/yinghork/socialiq_data/matchings/lam0.1
python matchings_to_roberta.py --type valid --lam 0.1 --dataset_path /work/yinghork/socialiq_data/matchings/valid --output_dir /work/yinghork/socialiq_data/matchings/lam0.1
```
- Run the following command with updated dataset paths and output directory: 
```
python train_roberta_multiple_choice.py --dataset_path /work/yinghork/socialiq_data/matchings/lam0.0 --dataset_name socialiq_permute_a2 --output_dir /work/yinghork/socialiq_data/matchings/lam0.0/matching_lr_1e-6_bs_3_unfreeze --learning_rate 1e-6 --batch_size 3
```

### 4. Get new matchings back in siq form for Merlot:
```
python matchings_to_siq.py --type train --lam 0.0 --dataset_path /work/yinghork/socialiq_data/matchings/train --output_dir /work/yinghork/socialiq_data
```

