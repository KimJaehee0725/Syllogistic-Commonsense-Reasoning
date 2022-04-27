import torch
import os
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class CustomSyllogismDataset(Dataset):
    def __init__(self, args, data_name, tokenizer, kfold_idx = None):
        self.data_path = os.path.join(args.data_path, data_name)
        self.raw_data = pd.read_csv(self.data_path, encoding = 'Windows-1252')
        self.raw_data = self.raw_data[self.raw_data["Syllogistic relation"] == "yes"]
        self.max_len = args.max_len
        self.batch_size = args.batch_size

        self.tokenizer = tokenizer
        self.bos = tokenizer.bos_token
        self.sep = tokenizer.sep_token
        self.eos = tokenizer.eos_token
        self.cls = tokenizer.cls_token
        self.mask = "<mask>"
        self.input_pad_id = tokenizer.pad_token_id
        self.target_pad_id = -100

        self.prem1 = self.raw_data["Premise 1"].to_list()
        self.prem2 = self.raw_data["Premise 2"].to_list()
        self.label = self.raw_data["Conclusion"].to_list()

        if kfold_idx is not None:
            self.prem1 = [sent for num, sent in enumerate(self.prem1) if num in kfold_idx]
            self.prem2 = [sent for num, sent in enumerate(self.prem2) if num in kfold_idx]
            self.label = [sent for num, sent in enumerate(self.label) if num in kfold_idx]

        assert len(self.prem1) == len(self.prem2) and len(self.prem2) == len(self.label),f"데이터 길이가 다름 \n Premise 1 : {len(self.prem1)} \n Premise 2 : {len(self.prem2)} \n Label : {len(self.label)}"

        self.encoder_text = [f"{self.bos}" + p1 + "." + f"{self.sep}" + p2 + "." + f"{self.eos}" + self.mask for p1, p2 in zip(self.prem1, self.prem2)]
        self.decoder_target = [f"{self.bos}" + l + "." + f"{self.eos}" for l in self.label]

    def __len__(self):
        return len(self.prem1)

    def __getitem__(self, idx):
        return self.__preprocess(self.encoder_text[idx], self.decoder_target[idx])

    def __preprocess(self, encoder_text, decoder_target):
        encoder_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
        decoder_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
        decoder_target_ids = torch.full((1, self.max_len), fill_value = self.target_pad_id)

        encoder_attn_mask = torch.zeros((1, self.max_len))
        decoder_attn_mask = torch.zeros((1, self.max_len))

        encoder_tokens = self.tokenizer.encode(encoder_text, add_special_tokens = False, return_tensors = 'pt')
        decoder_tokens = self.tokenizer.encode(decoder_target, add_special_tokens = False, return_tensors = 'pt')
        
        encoder_token_ids[0, :encoder_tokens.shape[1]] = encoder_tokens
        decoder_token_ids[0, :decoder_tokens.shape[1]-1] = decoder_tokens[:, :-1]
        decoder_target_ids[0, :decoder_tokens.shape[1]-1] = decoder_tokens[:, 1:]

        encoder_attn_mask[0, :encoder_tokens.shape[1]] = 1
        decoder_attn_mask[0, :decoder_tokens.shape[1]-1] = 1

        return encoder_token_ids, encoder_attn_mask, decoder_token_ids, decoder_attn_mask, decoder_target_ids

def collate_fn(batch) :
    encoder_ids = []
    encoder_attn = []
    
    decoder_ids = []
    decoder_attn = []
    decoder_targets = []
    
    for encoder_token_ids, encoder_attn_mask, decoder_token_ids, decoder_attn_mask, decoder_target_ids in batch:
        encoder_ids.append(encoder_token_ids)
        encoder_attn.append(encoder_attn_mask)
        decoder_ids.append(decoder_token_ids)
        decoder_attn.append(decoder_attn_mask)
        decoder_targets.append(decoder_target_ids)

    model_inputs = {
        "input_ids" : torch.cat(encoder_ids, dim = 0),
        "attention_mask" : torch.cat(encoder_attn, dim = 0),
        "decoder_input_ids" : torch.cat(decoder_ids, dim = 0),
        "decoder_attention_mask" : torch.cat(decoder_attn, dim = 0),
        "labels" : torch.cat(decoder_targets, dim = 0)
    }

    return model_inputs
