#! /usr/bin/env bash

cd ../.. > /dev/null

export PYTHONPATH=.:$PYTHONPATH

# download data, generate manifests
python Work/datasets/aishell_1_prepare.py \
--manifest_prefix='Work/aishell_1/data/manifest.aishell_1' \
--target_dir='~/data/ASR/aishell_1' \

if [ $? -ne 0 ]; then
    echo "Prepare Aishell failed. Terminated."
    exit 1
fi

# background data
python Work/datasets/noise_prepare.py \
--manifest_prefix='Work/noise/manifest.farfiled.background' \
--target_dir='~/data/_Farfiled_background_' \

if [ $? -ne 0 ]; then
    echo "Prepare Aishell failed. Terminated."
    exit 1
fi

echo "Manifest preparation done!"
exit 0