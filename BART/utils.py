import torch
import numpy as np
import random

from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_score(label : list, prediction:list):
    """
    label : list of string
    prediction : list of string

    return : bleu, rouge (dict)
    """
    label = label
    prediction = prediction
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
