#!/bin/bash
output_file=$1
epoch=$2
nohup python -u run.py sognn -d pubmed -nh 10 -e $epoch --plot_influence > $output_file 2>&1 &