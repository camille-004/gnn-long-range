#!/bin/bash
output_file=$3
epoch=$2
model=$1
dataset='pubmed'
nohup python -u run.py $model -d $dataset -nh 10 -e $epoch -a tanh --plot_influence --plot_energy > $output_file 2>&1 &