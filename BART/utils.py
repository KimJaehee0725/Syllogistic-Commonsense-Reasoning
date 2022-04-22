import torch
import numpy as np
import random

import datasets

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class Score_Calculator(self):
    def __init__(self, tokenizer) :
        self.bleu_scorer = datasets.load_metric('bleu')
        self.rouge_scorer = datasets.load_metric('rouge')
        self.special_tokens = [tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id]

    def update_score(self, label : torch.tensor, prediction : torch.tensor) :
        label = [[[token for token in sent if token not in self.special_tokens]] for sent in label]
        prediction = [[token for token in sent if token not in self.special_tokens] for sent in prediction]

        self.bleu_scorer.add_batch(references = label, predictions = prediction)
        self.rouge_scorer.add_batch(references = label, predictions = prediction)

    def get_final_score(self):
        bleu = self.bleu_scorer.compute()
        rouge = self.rouge_scorer.compute()

        bleu = {f'bleu-{i+1}' : bleu.precisions[i] for i in range(4)}
        rouge = {f'rouge-{i}' : rouge.[f"rouge{i}"] mid.fmeasure for i in ['1', '2', 'L']}
        return bleu, rouge


def get_score(label : torch.tensor, prediction:torch.tensor, tokenizer):
    """
    label : tensor of shape (batch_size, seq_len)
    prediction : tensor of shape (batch_size, seq_len)

    return : bleu, rouge, Bert Score (dict)
    """
    label = tokenizer.batch_decode(label, skip_special_tokens = True)
    prediction = tokenizer.batch_decode(prediction, skip_special_tokens = True)

    bleu = datasets.load_metric('bleu')
    rouge = datasets.load_metric('rouge')
    bert_score = datasets.load_metric('bertscore')




    bleu = {"bleu-1": 0, "bleu-2": 0, "bleu-3": 0, "bleu-4": 0}
    rouge = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    for i in range(len(label)):
        rouges = rouge_score([prediction[i]], [label[i]])
        bleu["bleu-1"] += bleu_score([prediction[i]], [label[i]], n_gram = 1)
        bleu["bleu-2"] += bleu_score([prediction[i]], [label[i]], 2)
        bleu["bleu-3"] += bleu_score([prediction[i]], [label[i]], 3)
        bleu["bleu-4"] += bleu_score([prediction[i]], [label[i]], 4)
        rouge["rouge-1"] += rouges["rouge1_fmeasure"]
        rouge["rouge-2"] += rouges["rouge2_fmeasure"]
        rouge["rouge-l"] += rouges["rougeL_fmeasure"]
    
    bleu["bleu-1"] /= len(label)
    bleu["bleu-2"] /= len(label)
    bleu["bleu-3"] /= len(label)
    bleu["bleu-4"] /= len(label)
    rouge["rouge-1"] /= len(label)
    rouge["rouge-2"] /= len(label)
    rouge["rouge-l"] /= len(label)
    return bleu, rouge

def save_model(model, fold_name, args, step):
    torch.save(model.state_dict(), args.model_path + f"BART_forGeneration_{fold_name}_step{step}.pt")
