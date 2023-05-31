#!/bin/bash
output_file=$4
epoch=$2
model=$1
dataset=$3
nohup python -u run.py $model -d $dataset -nh 1 -e $epoch -a relu -r 3 --plot_influence --plot_energy > $output_file 2>&1 &