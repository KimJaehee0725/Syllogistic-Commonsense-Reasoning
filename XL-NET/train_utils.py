import torch
import wandb
import logging

from tqdm import tqdm

from utils import get_score, save_model
logging.getLogger("lightning").setLevel(logging.CRITICAL)

def train(args, model, dataloader, optimizer, test_dataloader = None, scheduler = None) :
    wandb.init(
        project = "DSBA Syllogistic Inference - XLNet",
        name = "훈련 가능한지 확인 중")
    wandb.config.update(args)
    wandb.watch(model)

    torch.cuda.empty_cache()
    losses = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model.to(device)

    step = 0

    for num, epoch in enumerate(range(args.num_epochs)) :
        for (input_ids, token_type_ids, attn_mask), label  in dataloader :
            step += 1
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attn_mask = attn_mask.to(device)
            label = label.to(device)

            outputs = model(
                input_ids = input_ids, 
                token_type_ids = token_type_ids, 
                attention_mask = attn_mask, 
                labels = label)

            loss = outputs.loss
            losses.append(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                scheduler.step()

            wandb.log({"train_loss" : loss.item(), "epoch" : num, "steps" : step})


            if step % args.log_interval == 0 :
                print(f"Epoch : {num + 1} / {args.num_epochs}")
                print(f"Loss : {loss.item()}")
                print(f"Learning rate : {optimizer.param_groups[0]['lr']}")
                print("-"*50)

            if step % args.eval_interval == 0 :
                train_f1_mean, train_acc_mean = test(args, model, dataloader)
                wandb.log({"train_f1" : train_f1_mean, "train_acc" : train_acc_mean})
                print(f"train F1 : {train_f1_mean}")
                print(f"train Acc : {train_acc_mean}")
                
                if test_dataloader:
                    test_f1_mean, test_acc_mean = test(args, model, test_dataloader)
                    wandb.log({'test_f1' : test_f1_mean, 'test_acc' : test_acc_mean})
                    print(f"test F1 : {test_f1_mean}")
                    print(f"test Acc : {test_acc_mean}")

                print("-"*50)

            if step % args.save_interval == 0 :
                save_model(model, args, step)


def test(args, model, dataloader, return_result = False) :
    model.eval()
    
    f1_list = []
    acc_list = []
    prediction_list = []
    with torch.no_grad():
        for (input_ids, token_type_ids, attn_mask), label in tqdm(dataloader) :
            input_ids = input_ids.to("cuda")
            token_type_ids = token_type_ids.to("cuda")
            attn_mask = attn_mask.to("cuda")
            label = label.to("cuda")

            outputs = model(
                input_ids = input_ids, 
                token_type_ids = token_type_ids, 
                attention_mask = attn_mask, 
                labels = label)

            prediction = outputs.logits.detach().argmax(dim = 1).cpu().squeeze().tolist()
            prediction_list.extend(prediction)
            f1, acc = get_score(label.cpu().squeeze().tolist(), prediction)

            f1_list.append(f1)
            acc_list.append(acc)

    model.train()
    if not return_result :
        f1_mean = sum(f1_list) / len(f1_list)
        acc_mean = sum(acc_list) / len(acc_list)
        return f1_mean, acc_mean
        
    else:
        return prediction_list