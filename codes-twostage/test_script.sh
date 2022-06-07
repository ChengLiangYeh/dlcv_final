#!/bin/bash
python3 ./run_test.py --data ../Blood_data/train --csv ../Blood_data/train.csv --modelrec ./backup/resnet13.pth --modelpre ./backup/resnet-lam30.pth --conf 0.8 --recordtxtname ./tmp_compare.txt
#python3 ./run_test.py --data ../Blood_data/train --csv ../Blood_data/train.csv --modelrec ./backup/resnet13.pth --modelpre ./backup/resnet-data90-lam30.pth --conf 0.8 --recordtxtname ./tmp_compare.txt
#python3 ./run_test.py --data ../Blood_data/train --csv ../Blood_data/train.csv --modelrec ./backup/resnet19.pth --modelpre ./backup/resnet-data90-lam30.pth --conf 0.8 --recordtxtname ./tmp_compare.txt

#for i in {8..20}
#    do
#        python3 ./run_test.py --data ../Blood_data/train --csv ../Blood_data/train.csv --modelrec ./backup/resnet13.pth --modelpre ../codes/backup/resnet-ep$i.pth --conf 0.8
#    done 
