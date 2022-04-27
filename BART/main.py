from arg_parser import make_parser
from data import CustomSyllogismDataset, collate_fn
from train_utils import train, test
from utils import set_seed

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BartTokenizerFast, BartForConditionalGeneration

import os
import pandas as pd

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""
todo : 
2. bertscore 계산 할 코드 작성
3. (필요 시) Gradient Accumulation 적용
4. (필요 시) Mixed Precision 적용
"""
def main():
    print("current working directory : ", os.getcwd())
    print("current gpu device : ", torch.cuda.current_device())
    args = make_parser()
    set_seed(args.seed)      

    df = pd.read_csv(os.path.join(args.data_path, "Avicenna_train.csv"), encoding = 'Windows-1252')
    X = df[df["Syllogistic relation"] == "yes"]["Premise 1"].to_list()
    y = df[df["Syllogistic relation"] == "yes"]["Syllogistic relation"].to_list()  

    if args.kfold != 0 :
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits = args.kfold, shuffle = True, random_state = args.seed)

        

        kfold_split = [(train_idx, val_idx) for train_idx, val_idx in splitter.split(X, y)]
        train_idx, val_idx = kfold_split[args.kfold_idx]
    else:
        train_idx = list(range(len(X)))

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

    train_datasets = CustomSyllogismDataset(args, "Avicenna_train.csv", tokenizer, kfold_idx = train_idx)
    train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)

    if args.kfold != 0 :
        val_datasets = CustomSyllogismDataset(args, "Avicenna_test.csv", tokenizer, kfold_idx = val_idx)
        val_dataloader = DataLoader(val_datasets, batch_size = args.eval_batch_size, collate_fn = collate_fn, shuffle = False)
        train(args, model, train_dataloader, tokenizer, f"fold-{args.kfold_idx}", val_dataloader)
    else:
        test_dataset = CustomSyllogismDataset(args, "Avicenna_test.csv", tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size = args.eval_batch_size, collate_fn = collate_fn, shuffle = False) 
        train(args, model, train_dataloader, tokenizer, "final train", test_dataloader)

    if args.save_final_model:
        model.save_pretrained(args.model_path)
        tokenizer.save_pretrained(args.model_path)

if __name__ == "__main__":
    main()
