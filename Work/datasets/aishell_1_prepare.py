# coding:utf-8
#
# Date : 2020.04.01
# Author: phecda-xu < >
#
# DEC:
#     Datasets about aishell_1 download and generate manifest

import os
import json
import codecs
import argparse
import soundfile
from tqdm import tqdm
from data_utils.utils import download, unpack

DATA_HOME = '~/datadisk/phecda/ASR'

URL_ROOT = 'http://www.openslr.org/resources/33'
DATA_URL = URL_ROOT + '/data_aishell.tgz'
MD5_DATA = 'f6bf18f56e2315d1fed4ac7eaf911582'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/aishell_1",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="Work/aishell_1/data/manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    if not os.path.exists(os.path.dirname(manifest_path_prefix)):
        os.makedirs(os.path.dirname(manifest_path_prefix))
    data_sets = ['train', 'dev', 'test']
    for data_set in data_sets:
        del json_lines[:]
        audio_dir = os.path.join(data_dir, 'wav', data_set)
        for subfolder, _, filelist in tqdm(sorted(os.walk(audio_dir))):
            for fname in filelist:
                audio_path = os.path.join(subfolder, fname)
                audio_data, samplerate = soundfile.read(audio_path)
                duration = float(len(audio_data) / samplerate)
                json_lines.append(
                    json.dumps(
                        {
                            'audio_filepath': audio_path,
                            'duration': duration,
                        },
                        ensure_ascii=False))
        manifest_path = manifest_path_prefix + '.' + data_set
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')


def prepare_dataset(url, md5sum, target_dir, manifest_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'data_aishell')
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
        # unpack all audio tar files
        audio_dir = os.path.join(data_dir, 'wav')
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for ftar in filelist:
                unpack(os.path.join(subfolder, ftar), subfolder, True)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    create_manifest(data_dir, manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        url=DATA_URL,
        md5sum=MD5_DATA,
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix)


if __name__ == '__main__':
    main()
