#! /usr/bin/env bash

cd ../.. > /dev/null

export PYTHONPATH=.:$PYTHONPATH

# download data, generate manifests
python Work/datasets/voicebank.py \
--manifest-prefix='Work/voicebank/data/manifest.voicebank' \
--target-dir='~/data/SE/voicebank' \
--sample-rate=8000

if [ $? -ne 0 ]; then
    echo "Prepare voicebank failed. Terminated."
    exit 1
fi