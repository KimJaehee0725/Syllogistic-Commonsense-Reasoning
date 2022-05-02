import torch
from torch.optim import AdamW

import gc
import os
import wandb
from tqdm import tqdm
import pandas as pd

from utils import Score_Calculator, save_model, save_generation, log_validation, test



def train(args, model, dataloader, tokenizer, fold_name = None, val_dataloader = None, scheduler = None) :
    print("="*20, f"Fold : {fold_name}", "="*20)
    
    wandb.init(
        project = args.project_name,
        name = fold_name, 
        reinit = True,
        group = args.group_name)
        
    wandb.config.update(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = AdamW(model.parameters(), lr = args.learning_rate)
    model.train()
    model.to(device)

    step = 0
    # if args.kfold != 0 :
    #     log_validation(args, model, val_dataloader, tokenizer, wandb, step, fold_name)

    for num, epoch in enumerate(range(args.num_epochs)) :
        print("-"*15, f"Epoch : {epoch}/ {args.num_epochs}", "-"*15)
        print("-"*15, f"Train", "-"*15)
        for num_iter, model_inputs in enumerate(tqdm(dataloader)) :
            model_inputs = {key : value.to(device) for key, value in model_inputs.items()}
            outputs = model(**model_inputs)

            loss = outputs.loss/args.accumulation_steps
            loss.backward()

            if (num_iter+1) % args.accumulation_steps == 0 :
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                loss = loss.detach().item()
                if scheduler:
                    scheduler.step()

                wandb.log({"train_loss" : loss, "epoch" : num, "steps" : step})


            if step % args.eval_interval == 0 :
                print("-"*15, f"{step} step train metric 계산 시작", "-"*15)
                print(f"Epoch : {num + 1} / {args.num_epochs}")
                print(f"Loss : {loss}")
                print(f"Learning rate : {optimizer.param_groups[0]['lr']}")
                print("-"*45)

            if (step % args.eval_interval == 0) and (step % args.accumulation_steps == 0) and (args.kfold != 0 ):
                print("-"*15, f"Epoch : {num} Validation 시작", "-"*15)
                log_validation(args, model, val_dataloader, tokenizer, wandb, step, fold_name)                    
                    
            if step % args.save_interval == 0 :
                save_model(model, args, fold_name = fold_name, step = step)

    log_validation(args, model, val_dataloader, tokenizer, wandb, step, fold_name, is_final = True)
    save_model(model, args, is_final = True)


