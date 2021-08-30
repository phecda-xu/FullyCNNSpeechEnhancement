#! /usr/bin/env bash

cd ../.. > /dev/null

export PYTHONPATH=.:$PYTHONPATH

# download data, generate manifests
python Work/datasets/aishell_2_prepare.py \
--manifest-prefix='Work/aishell_2/data/manifest.aishell_2' \
--target-dir='~/data/ASR/AISHELL-2' \
--sample-rate=8000

if [ $? -ne 0 ]; then
    echo "Prepare Aishell_2 failed. Terminated."
    exit 1
fi

# background data
python Work/datasets/noise_prepare.py \
--manifest-prefix='Work/noise/manifest.farfiled.background' \
--target-dir='~/data/Noise/_Farfiled_background_' \
--sample-rate=8000

if [ $? -ne 0 ]; then
    echo "Prepare Noise failed. Terminated."
    exit 1
fi

echo "Manifest preparation done!"
exit 0