#!/bin/bash

WORKSPACE="workspace"
mkdir $WORKSPACE
TR_SPEECH_DIR="../../data/enhanceData/train_speech"
TR_NOISE_DIR="../../data/enhanceData/train_noise"
TE_SPEECH_DIR="../../data/enhanceData/test_speech"
TE_NOISE_DIR="../../data/enhanceData/test_noise"

# Create mixture csv.
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --magnification=2
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test  --magnification=2


# Calculate mixture features.
TR_SNR=0
TE_SNR=0
python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR
python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR

# Pack features.
N_CONCAT=7
N_HOP=3
python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP
python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --snr=$TE_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP
