# coding:utf-8
#
#
#
#

import os
import time
import logging
import argparse
from config import load_conf_info
from model_utils.trainer import FullyCNNTrainer
from data_utils.data_loader import DataLoader, Sampler, DataSet


def main(config):

    logging_path = config.get("training", "log_dir")
    epochs = int(config.get("training", 'epochs'))
    batch_size = int(config.get("training", 'batch_size'))

    net_arch = config.get("model", 'net_arch')
    net_work = config.get("model", 'net_work')

    window_ms = int(config.get("data", "window_ms"))
    stride_ms = int(config.get("data", "stride_ms"))
    sample_rate = int(config.get("data", 'sample_rate'))
    train_clean_manifest = config.get("data", "train_manifest_path")
    val_clean_manifest = config.get("data", "val_manifest_path")
    train_noise_manifest = config.get("data", "train_noise_manifest")
    val_noise_manifest = config.get("data", "val_manifest_path")
    snr = float(config.get("data", "snr"))
    # logging INFO
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    log_name = time.asctime().replace(':', "-").replace(" ", "_")
    logger = logging.getLogger("train.py")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("{}/{}_{}_{}_log.txt".format(logging_path,
                                                               net_arch, net_work, log_name), mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    train_dataset = DataSet(manifest_filepath=train_clean_manifest,
                            noise_manifest=train_noise_manifest,
                            sample_rate=sample_rate,
                            window_ms=window_ms,
                            stride_ms=stride_ms,
                            snr=snr)
    val_dataset = DataSet(manifest_filepath=val_clean_manifest,
                          noise_manifest=val_noise_manifest,
                          sample_rate=sample_rate,
                          window_ms=window_ms,
                          stride_ms=stride_ms,
                          snr=snr,
                          use_complex=True)

    train_sampler = Sampler(train_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size, train_sampler)

    val_loader = DataLoader(val_dataset, batch_size, sampler=None)

    SE_Trainer = FullyCNNTrainer(config)
    SE_Trainer.train(train_loader, val_loader, epochs, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', default='', type=str, help='cfg file for train')
    args = parser.parse_args()
    train_config = load_conf_info(args.cfg)
    main(train_config)
