#!/bin/bash
python3 ./run_train.py --data ../Blood_data/train --csv ../Blood_data/train.csv --batch 48 --epoch 20 --start 0 --save ./backup/ --lam 0.3

# 1, f2 57
# 2, f2 71   , positive loss weight lam = 0.8
# 3, f2 70   , positive loss weight lam = 0.6 (precision 56 better than #13)
# 4, f2 70   , normalize with mean=0.1592, std=0.0651, more stable(?)
# 5, f2 72.18, random gaussian for augmentation
# 6, f2 70   , positive loss weight lam = 0.7
# 7, f2 70.58, no center crop
# 8, f2 72.04, lam=0.8, random gaussian, normalize, no center crop
# 9, f2 69.94, loss reweighting on positive and perclass, normalize (precision 52)
#10, f2 71.97, random affine based on #5
#11, f2 72.11, random vertical flip based on #5
#12, f2 70   , RMSprop based on #5
#13, f2 73.69, all augmentation (precision 50) rotation 15, affine degree 15 (just degree)
#14, f2 64   , lam = 0.4 (use post processing, sequential data characteristic) (precision 78, and recall 61
#15, f2      , all augmentation like #13, add affine parameters like translation... etc. and '''rotation 30'''
#16, f2 72.35, all aug, add affine params, but '''rotation 15'''
#17  f2 72.85, #13 every epoch load in twice of class "edh" data
#18, f2 67.67, lam = 0.3 (use post processing, sequential data characteristic)[val set]: Precision = 71.49%   Recall = 66.78% f2-score = 67.67%
#backup-data90
#19, f2 73.72, more training data
#20, f2      , lam = 0.3 + class weight
#backup-data90-lam30
#21, f2 67.61, lam = 0.3, more training data (without affine)
#22, f2 73.21, lam = 0.8, more training data (without affine)
#backup
#23, f2      , lam = 0.3, more training data
