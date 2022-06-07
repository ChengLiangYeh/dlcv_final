#!/bin/bash
python3 ./codes-twostage/run_predict.py --data $1 --modelrec ./model2.pth --modelpre ./model1.pth --out $2
