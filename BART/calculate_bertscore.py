import datasets
import pandas as pd
import numpy as np
# class BertScore_Calculator():
#     def __init__(self, tokenizer) :
#         self.bert_scorer = datasets.load_metric('bertscore')
#         self.special_tokens = [tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id]

#     def update_score(self, label : list, prediction : list) :
#         """
#         input 
#         =========
#         label : list of sentences as one mini-batch
#         prediction : list of sentences as one mini-batch

#         return 
#         =========
#         None
#         """
#         self.bert_scorer.add_batch(references = label, predictions = prediction)

def main():
    data_path = "/project/Syllogistic-Commonsense-Reasoning/BART/generation_log/final_generation.csv"
    
    bert_scorer = datasets.load_metric('bertscore')
    
    data = pd.read_csv(data_path)
    label = data['label'].tolist()
    generation = data['generation'].tolist()

    score = bert_scorer.compute(
        references = label, 
        predictions = generation, 
        lang = 'en', 
        verbose = True
        )

    print(
        f"""
        BERT score
        ===========
        precision : {np.mean(score["precision"])}
        recall : {np.mean(score["recall"])}
        f1 : {np.mean(score["f1"])}
        """
    )

if __name__ == "__main__":
    main()