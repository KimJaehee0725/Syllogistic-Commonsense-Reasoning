import torch
import wandb

from tqdm import tqdm
import pandas as pd

from utils import get_score, save_model

from torch.optim import AdamW
import gc

def train(args, model, dataloader, tokenizer, fold_name = None, val_dataloader = None, scheduler = None) :
    if fold_name == None:
        fold_name = "not K-Fold Validation"
    else:
        print("-"*50, f"Fold : {fold_name}", "-"*50)
    wandb.init(
        project = args.project_name,
        name = fold_name, 
        reinit = True)
    wandb.config.update(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = AdamW(model.parameters(), lr = args.learning_rate)
    model.train()
    model.to(device)

    step = 0
    for num, epoch in enumerate(range(args.num_epochs)) :
        for model_inputs in dataloader :
            step += 1
            model_inputs = {key : value.to(device) for key, value in model_inputs.items()}
            outputs = model(**model_inputs)

            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                scheduler.step()

            wandb.log({"train_loss" : loss.detach().item(), "epoch" : num, "steps" : step})


            if step % args.log_interval == 0 :
                print("-"*50, f"{step} step train metric 계산 시작", "-"*50)
                print(f"Epoch : {num + 1} / {args.num_epochs}")
                print(f"Loss : {loss.detach().item()}")
                print(f"Learning rate : {optimizer.param_groups[0]['lr']}")
                print("-"*50)

            if step % args.eval_interval == 0 :
                print("-"*50, f"Epoch : {num} Validation 시작", "-"*50)
                # train_bleu, train_rouge, train_generation, train_loss = test(args, model, dataloader, tokenizer)
                # wandb.log({
                #     f"train bleu-{i}" : train_bleu[f'bleu-{i}'] for i in range(1, 5)
                # })
                # wandb.log({
                #     f"train rouge-{i}" : train_rouge[f'rouge-{i}'] for i in [1, 2, 'l']
                # })
                # print(f"train bleu-1 : {train_bleu['bleu-1']}")
                # print(f"train rouge-L : {train_rouge['rouge-l']}")
                
                if val_dataloader : 
                    val_bleu, val_rouge, val_generation, val_loss = test(args, model, val_dataloader, tokenizer)
                    wandb.log({
                        f"valid bleu-{i}" : val_bleu[f'bleu-{i}'] for i in range(1, 5)    
                    })
                    wandb.log({
                        f"valid rouge-{i}" : val_rouge[f'rouge-{i}'] for i in [1, 2, 'l']
                    })
                    wandb.log({"valid_loss" : val_loss})
                    save_generation(f"{args.generation_path}/val_generation_{step}-steps_{fold_name}.csv", val_generation)
                    print(f"valid bleu-1 : {val_bleu['bleu-1']}")
                    print(f"valid rouge-L : {val_rouge['rouge-l']}")
                    print("-"*50)
                    
            if step % args.save_interval == 0 :
                save_model(model, fold_name, args, step)

def test(args, model, dataloader, tokenizer, return_result = False) :
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bleu_list = []
    rouge_list = []
    prediction_list = []
    label_list = []
    input_list = []
    loss_list = []

    with torch.no_grad():
        for model_inputs in tqdm(dataloader) :
            model_inputs = {key : value.to(device) for key, value in model_inputs.items()}
            outputs = model(**model_inputs)
            labels = model_inputs["labels"].cpu().squeeze().tolist()
            labels = [[token for token in label if token != -100] for label in labels]
            loss = outputs.loss.detach().item()
            loss_list.append(loss)

            prediction = model.generate(model_inputs["input_ids"], max_length = 128).cpu()
            prediction_str_batch = tokenizer.batch_decode(prediction, skip_special_tokens = True)
            label_str_batch = tokenizer.batch_decode(labels, skip_special_tokens = True)
            input_str_batch = tokenizer.batch_decode(model_inputs["input_ids"].cpu().tolist(), skip_special_tokens = True)


            input_list.extend(input_str_batch)
            prediction_list.extend(prediction_str_batch)
            label_list.extend(label_str_batch)
            bleu, rouge = get_score(label_str_batch, prediction_str_batch)
            bleu_list.append(bleu)
            rouge_list.append(rouge)

    model.train()
    bleu_df = pd.DataFrame(bleu_list)
    rouge_df = pd.DataFrame(rouge_list)

    bleu_mean = bleu_df.mean(axis = 0).to_dict()
    rouge_mean = rouge_df.mean(axis = 0).to_dict()

    generation_df = pd.DataFrame({
        "input" : input_list,
        "generation" : prediction_list,
        "label" : label_list})
    
    valid_loss = sum(loss_list)/len(dataloader)

    del bleu_list, rouge_list, prediction_list, label_list, input_list, loss_list, bleu_df, rouge_df, model_inputs, outputs, labels, prediction, label_str_batch, input_str_batch, loss, prediction_str_batch
    gc.collect()
    torch.cuda.empty_cache()
    
    return bleu_mean, rouge_mean, generation_df, valid_loss

def save_generation(path, file):
    file.to_csv(path, index = False)