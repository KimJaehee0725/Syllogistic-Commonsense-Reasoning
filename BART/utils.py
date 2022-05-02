import torch
import numpy as np
import random
import os

import datasets
import nltk.translate.bleu_score as bleu
import tensorflow as tf
import tensorflow_text as text
from numpy import mean

from tqdm import tqdm
import pandas as pd
import gc


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

"""
calculator = Score_Calculator(tokenizer)
for batch in test_dataloader:
    generation = model.generate(**batch)
    labels = batch["labels"]

    calculator.update_score(labels, generation)
bleu, rouge = calculator.get_final_score()
"""
class Score_Calculator():
    def __init__(self, tokenizer) :
        self.rouge_scorer = text.metrics.rouge_l
        self.rouge_score = 0

        self.bleu_scorer = bleu.corpus_bleu
        self.bleu_score = 0

        self.tokenizer = tokenizer
        self.special_tokens = [tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id]
        
        self.count = 0
    def update_score(self, label : torch.tensor, prediction : torch.tensor) :
        bleu_label = [[[self.tokenizer.decode(token) for token in sent if token not in self.special_tokens]] for sent in label]
        bleu_prediction = [[self.tokenizer.decode(token) for token in sent if token not in self.special_tokens] for sent in prediction]
        self.bleu_score += self.bleu_scorer(bleu_label, bleu_prediction)

        rouge_label = [tf.ragged.constant([[self.tokenizer.decode(token) for token in sent if token not in self.special_tokens]]) for sent in label]
        rouge_prediction = [tf.ragged.constant([[self.tokenizer.decode(token) for token in sent if token not in self.special_tokens]]) for sent in prediction]
        self.rouge_score += mean([self.rouge_scorer(label, prediction).f_measure.numpy()[0] for label, prediction in zip(rouge_label, rouge_prediction)])
        
        self.count += 1 
    def get_final_score(self):
        return self.bleu_score/self.count, self.rouge_score/self.count



def save_generation(file, args, fold_name = None, step = None, is_final = False) :
    if is_final:
        file.to_csv(os.path.join(args.generation_path, args.datetime, f"final_generation.csv"), index = False)
    else :
        file.to_csv(os.path.join(args.generation_path, args.datetime, f"val_generation_{step}-steps_{fold_name}.csv"), index = False)

def save_model(model, args, fold_name = None, step = None, is_final = False):
    if is_final:
        torch.save(model.state_dict(), os.path.join(args.model_path, args.datetime, f"BARTforGeneration_final.pt"))
    else:
        torch.save(model.state_dict(), os.path.join(args.model_path, args.datetime, f"BARTforGeneration_{fold_name}_step{step}.pt"))

def log_validation(args, model, val_dataloader, tokenizer, wandb, step, fold_name, is_final = False) :
    val_bleu, val_rouge, val_generation, val_loss = test(args, model, val_dataloader, tokenizer)
    wandb.log({"valid bleu" : val_bleu})
    wandb.log({"valid rouge" : val_rouge})
    wandb.log({"valid_loss" : val_loss})

    save_generation(val_generation, args, fold_name = fold_name, step = step, is_final = is_final)

    print(f"valid bleu : {val_bleu}")
    print(f"valid rouge : {val_rouge}")
    print(f"valid loss : {val_loss}")
    print("-"*50)

def test(args, model, dataloader, tokenizer, return_result = False) :
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prediction_list = []
    label_list = []
    input_list = []
    loss_list = []

    scorer = Score_Calculator(tokenizer)

    with torch.no_grad():
        print("-"*15, f"Test", "-"*15)
        for model_inputs in tqdm(dataloader) :
            model_inputs = {key : value.to(device) for key, value in model_inputs.items()}

            # calculate loss
            outputs = model(**model_inputs)
            loss = outputs.loss.detach().item()
            loss_list.append(loss)

            # calculate bleu, rouge score
            prediction = model.generate(model_inputs["input_ids"], max_length = 128).cpu()  
            labels = model_inputs["labels"].cpu().squeeze().tolist()
            labels = [[token for token in label if token != -100] for label in labels]
            scorer.update_score(labels, prediction)

            # save generation & label string
            input_str_batch = tokenizer.batch_decode(model_inputs["input_ids"].cpu().tolist(), skip_special_tokens = True)
            input_list.extend(input_str_batch)

            prediction_str_batch = tokenizer.batch_decode(prediction.tolist(), skip_special_tokens = True)
            prediction_list.extend(prediction_str_batch)

            label_str_batch = tokenizer.batch_decode(labels, skip_special_tokens = True)
            label_list.extend(label_str_batch)

    model.train()

    generation_df = pd.DataFrame({
        "input" : input_list,
        "generation" : prediction_list,
        "label" : label_list})
    
    valid_loss = sum(loss_list)/len(dataloader)

    bleu, rouge = scorer.get_final_score()
    del scorer, model_inputs, outputs, loss_list, prediction_list, label_list, input_list
    gc.collect()
    torch.cuda.empty_cache()
    
    return bleu, rouge, generation_df, valid_loss
