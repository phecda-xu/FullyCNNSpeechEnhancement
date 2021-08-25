#!/usr/bin/env bash

cd ../.. > /dev/null

CUDA_VISIBLE_DEVICES=0 \
python -u train.py --cfg Work/aishell_1/cfg/fully_cnn_train.cfg
