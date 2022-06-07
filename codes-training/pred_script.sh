#!/bin/bash
python3 ./run_predict.py --data ../Blood_data/test --batch 48 --model ./backup-model/resnet13.pth --conf 0.8 --out ../pred.csv
