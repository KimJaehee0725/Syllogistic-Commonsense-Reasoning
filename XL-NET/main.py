from arg_parser import make_parser
from data import CustomSyllogismDataset, collate_fn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import XLNetTokenizer, XLNetForSequenceClassification

from utils import set_seed

from train_utils import train, test

def main():
    args = make_parser()

    set_seed(args.seed)
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels = 2)
    
    train_datasets = CustomSyllogismDataset(args, "Avicenna_train.csv", tokenizer)
    train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)

    test_datasets = CustomSyllogismDataset(args, "Avicenna_test.csv", tokenizer)
    test_dataloader = DataLoader(test_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = False)

    optimizer = AdamW(model.parameters(), lr = args.learning_rate)
    train(args, model, train_dataloader, optimizer, test_dataloader)
    f1_mean, acc_mean = test(args, model, test_dataloader)
    print(f"Test F1 : {f1_mean}")
    print(f"Test Acc : {acc_mean}")
    print("-"*50)

if __name__ == "__main__":
    main()
