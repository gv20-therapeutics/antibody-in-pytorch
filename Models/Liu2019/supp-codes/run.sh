#!/bin/bash
mkdir log
#cp -r ../data/seq_gen ../results/seq_gen
bash propose_seq.sh
python postprocess.py
python plot2.py