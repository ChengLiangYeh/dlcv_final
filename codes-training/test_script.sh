#!/bin/bash
python3 ./run_test.py --data ../Blood_data/train --csv ../Blood_data/train.csv --batch 48 --model ./backup-model/resnet1.pth --conf 0.8

#for i in {0..15}
#    do
#        python3 ./run_test.py --data ../Blood_data/train --csv ../Blood_data/train.csv --batch 32 --model ./backup/resnet-ep$i.pth --conf 0.9
#    done
