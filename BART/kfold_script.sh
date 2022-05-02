#!/bin/bash



for fold_idx in {0..4}
do
 python3 main.py -group_name KFOLD_VALIDATION -kfold_idx $fold_idx -num_epochs 10
done

