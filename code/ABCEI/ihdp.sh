#!/bin/bash

mkdir results
mkdir results/ihdp

python abcei_param_search.py configs/ihdp.txt 10

python evaluate.py configs/ihdp.txt 1
