#!/bin/bash



for fold_idx in {0..4}
do
 python3 main.py -kfold_idx $fold_idx
done

