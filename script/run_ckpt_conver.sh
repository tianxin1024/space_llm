#!/bin/bash

python ./ckpt_convert.py \
        -head_num 16 \
        -i /home/tianxin/data/LLM/llama2/ \
        -o ../models/llama-models/c-model/345m/ \
        -t_g 1 \
        -i_g 1 \
        --vocab-path ../models/gpt2-vocab.json \
        --merges-path ../models/gpt2-merges.txt
