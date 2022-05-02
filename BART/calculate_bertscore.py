import datasets
import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Calculate BERT Score')
    parser.add_argument(
        '-data_path',
        type=str,
        help = "generation log 위치")

    parser = parser.parse_args()

    data_path = os.path.join("/project/Syllogistic-Commonsense-Reasoning/BART/generation_log/", parser.data_path, "final_generation.csv")
    
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