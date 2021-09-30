# coding: utf-8
#
#
#
#
import os
import argparse
from config import load_conf_info
from model_utils.tester import FullyCNNTester
from data_utils.data_loader import DataLoader, DataSet


def main(config, num_works):
    window_ms = int(config.get("data", "window_ms"))
    stride_ms = int(config.get("data", "stride_ms"))
    sample_rate = int(config.get("data", 'sample_rate'))
    test_manifest = config.get("data", "test_manifest_path")
    test_noise_manifest = config.get("data", "test_noise_manifest") if config.has_option("data", "test_noise_manifest") else None
    batch_size = int(config.get("testing", "batch_size"))
    snr = float(config.get("data", "snr"))

    test_dataset = DataSet(manifest_filepath=test_manifest,
                           noise_manifest=test_noise_manifest,
                           sample_rate=sample_rate,
                           window_ms=window_ms,
                           stride_ms=stride_ms,
                           snr=snr,
                           use_complex=True)

    test_loader = DataLoader(test_dataset, batch_size, sampler=None, num_works=num_works)
    SE_Tester = FullyCNNTester(config)
    SE_Tester.test(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', default='', type=str, help='cfg file for test')
    parser.add_argument('--num-works', default=16, type=int, help='multi thread for data_loader')
    args = parser.parse_args()
    test_config = load_conf_info(args.cfg)
    main(test_config, args.num_works)
