import torch
import numpy as np
import random

from sklearn.metrics import f1_score, accuracy_score


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_score(label, prediction):
    # 성능 평가하기
    f1 = f1_score(label, prediction, average="macro")
    acc = accuracy_score(label, prediction)

    return f1, acc

def save_model(model, args, step):
    torch.save(model.state_dict(), args.model_path + f"xlnet_step_{step}.pt")
