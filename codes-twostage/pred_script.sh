#!/bin/bash
python3 ./run_predict.py --data ../Blood_data/test --modelrec ./backup/resnet13.pth --modelpre ./backup/resnet-lam30.pth --conf 0.8 --out ../pred.csv
