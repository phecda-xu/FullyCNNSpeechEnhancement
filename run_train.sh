#!/bin/bash
# Train.
WORKSPACE="workspace"
TR_SNR=0
TE_SNR=0
LEARNING_RATE=1e-4
CUDA_VISIBLE_DEVICES=0,1 python train.py train --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --lr=$LEARNING_RATE
