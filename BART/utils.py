import torch
import numpy as np
import random
import os

import datasets

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

class Score_Calculator():
    def __init__(self, tokenizer) :
        self.bleu_scorer = datasets.load_metric('bleu')
        self.rouge_scorer = datasets.load_metric('rouge')
        self.tokenizer = tokenizer
        self.special_tokens = [tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id]

    def update_score(self, label : torch.tensor, prediction : torch.tensor) :
        bleu_label = [[[token for token in sent if token not in self.special_tokens]] for sent in label]
        bleu_prediction = [[token for token in sent if token not in self.special_tokens] for sent in prediction]

        rouge_label = self.tokenizer.batch_decode(label, skip_special_tokens = True)
        rouge_prediction = self.tokenizer.batch_decode(prediction, skip_special_tokens = True)

        self.bleu_scorer.add_batch(references = bleu_label, predictions = bleu_prediction)
        self.rouge_scorer.add_batch(references = rouge_label, predictions = rouge_prediction)

    def get_final_score(self):
        bleu = self.bleu_scorer.compute()
        rouge = self.rouge_scorer.compute()

        bleu = {f'bleu-{i+1}' : bleu["precisions"][i] for i in range(4)}
        rouge = {f'rouge-{i}' : rouge[f"rouge{i}"].mid.fmeasure for i in ['1', '2', 'L']}
        return bleu, rouge

def save_model(model, fold_name, args, step):
    torch.save(model.state_dict(), args.model_path + f"BART_forGeneration_{fold_name}_step{step}.pt")
