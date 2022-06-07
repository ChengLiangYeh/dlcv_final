#!/bin/bash
http1='https://www.dropbox.com/s/ojbi33h7spfnhcn/resnet-lam30.pth?dl=1'
wget -O ./model1.pth $http1
http2='https://www.dropbox.com/s/6w70e59m09i3r9p/resnet13.pth?dl=1'
wget -O ./model2.pth $http2
