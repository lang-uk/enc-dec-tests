#!/bin/bash
NUM_GPUS=2

for gpu_id in $(seq 1 $((NUM_GPUS))); do
    CUDA_VISIBLE_DEVICES=$gpu_id python optuna-opus.py train \
        --data-path data/pararook.jsonl \
        --output-dir ./exps/optuna \
        --n-trials 20 \
        --n-jobs 1 \
        --num-epochs 1 \
        --storage "sqlite:///optuna_study.db" &
done
wait