#!/usr/bin/env bash

cd ../.. > /dev/null

CUDA_VISIBLE_DEVICES='' \
python -u test.py --cfg Work/aishell_2/cfg/fully_cnn_test.cfg --num-works 1
