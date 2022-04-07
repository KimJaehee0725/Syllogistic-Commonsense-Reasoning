import torch
import os
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class CustomSyllogismDataset(Dataset):
    def __init__(self, args, data_name, tokenizer):
        self.data_path = os.path.join(args.data_path, data_name)
        self.raw_data = pd.read_csv(self.data_path, encoding = 'Windows-1252')
        self.max_len = args.max_len
        self.batch_size = args.batch_size

        self.tokenizer = tokenizer
        self.bos = tokenizer.bos_token_id
        self.sep = tokenizer.sep_token_id
        self.eos = tokenizer.eos_token_id
        self.cls = tokenizer.cls_token_id
        self.pad = tokenizer.pad_token_id

        self.label_map = {
            "yes" : 1,
            "no" : 0
        }

        self.prem1 = self.raw_data["Premise 1"]
        self.prem2 = self.raw_data["Premise 2"]
        self.label = self.raw_data["Syllogistic relation"]

        assert len(self.prem1) == len(self.prem2) and len(self.prem2) == len(self.label),f"데이터 길이가 다름 \n Premise 1 : {len(self.prem1)} \n Premise 2 : {len(self.prem2)} \n Label : {len(self.label)}"

    def __len__(self):
        return len(self.prem1)

    def __getitem__(self, idx):
        return self.__preprocess(self.prem1[idx], self.prem2[idx], self.label[idx])

    def __preprocess(self, prem1, prem2, label):
        token_ids = torch.full((1, self.max_len), fill_value = self.pad)

        prem1_tokens = self.tokenizer.encode(prem1, return_tensors = "pt")[0, :-1]
        prem2_tokens = self.tokenizer.encode(prem2, return_tensors = "pt")[0, :]
        label = self.label_map[label]

        prem1_len = prem1_tokens.shape[0]
        prem2_len = prem2_tokens.shape[0]
        total_len = prem1_len + prem2_len 

        diff = self.max_len - total_len
        if diff < 0 :
            prem2_tokens = prem2_tokens[:-diff]
            prem2_len = prem2_tokens.shape[0]
            total_len = self.max_len

        token_ids[0,:prem1_len] = prem1_tokens
        token_ids[0,prem1_len:total_len] = prem2_tokens

        diff = 0 if diff < 0 else diff
        token_type_ids = torch.tensor([0]*prem1_len + [1]*prem2_len + [2]*diff)
        attn_mask = torch.tensor([1]*total_len + [0]*diff)

        return token_ids, token_type_ids, attn_mask, label

def collate_fn(batch) :
    input_ids = []
    token_type_ids = []
    attn_mask_ids = []
    labels = []

    for token_ids, token_type_id, attn_mask, label in batch:
        input_ids.append(token_ids)
        token_type_ids.append(token_type_id.unsqueeze(0))
        attn_mask_ids.append(attn_mask.unsqueeze(0))
        labels.append(label)

    input_ids = torch.cat(input_ids)
    token_type_ids = torch.cat(token_type_ids)
    attn_mask_ids = torch.cat(attn_mask_ids)
    labels = torch.tensor(labels).unsqueeze(0)
    return (input_ids, token_type_ids, attn_mask_ids), labels

