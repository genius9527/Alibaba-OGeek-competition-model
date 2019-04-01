#!/bin/sh

unzip $(ls|grep rank81_0.7468_cpu.zip)

echo $(pwd)

sh run.sh ./train.txt ./vali.txt ./test.txt 
python eval.py
