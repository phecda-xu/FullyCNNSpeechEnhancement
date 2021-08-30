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
import resampy
import argparse
import soundfile as sf
from tqdm import tqdm
from data_utils.utils import download, unpack
from multiprocessing import Process, Queue, Pool, Manager

DATA_HOME = '~/datadisk/phecda/ASR'

URL_ROOT = 'http://www.openslr.org/resources/33'
DATA_URL = URL_ROOT + '/data_aishell.tgz'
MD5_DATA = 'f6bf18f56e2315d1fed4ac7eaf911582'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target-dir",
    default=DATA_HOME + "/aishell_1",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest-prefix",
    default="Work/aishell_1/data/manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
parser.add_argument(
    "--sample-rate",
    default=16000,
    type=int,
    help="target audio sample rate")
args = parser.parse_args()


def load_and_resample(audio_path, n):
    path_dic = {"8000": "8K", "32000": "32K"}
    audio_data, samplerate = sf.read(audio_path)
    if len(audio_data) < 100:
        return None
    if samplerate != args.sample_rate:
        audio_data = resampy.resample(
            audio_data, samplerate, args.sample_rate, filter='kaiser_best')
        samplerate = args.sample_rate
        audio_path = audio_path.replace("ASR", "{}ASR".format(path_dic[str(args.sample_rate)]))
        try:
            if not os.path.exists(os.path.dirname(audio_path)):
                os.makedirs(os.path.dirname(audio_path))
        except:
            print('Skip dir error...')
        sf.write(audio_path, audio_data, samplerate)
    duration = float(len(audio_data) / samplerate)
    json_str = json.dumps(
        {
            'audio_filepath': audio_path,
            'duration': duration,
        },
        ensure_ascii=False)
    # print("pid : {} ; process audio :{}".format(os.getpid(), n))
    return json_str


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    if not os.path.exists(os.path.dirname(manifest_path_prefix)):
        os.makedirs(os.path.dirname(manifest_path_prefix))
    data_sets = ['train', 'dev', 'test']
    for data_set in data_sets:
        json_lines = []
        pool = Pool()
        results = []
        n = 0
        audio_dir = os.path.join(data_dir, 'wav', data_set)
        for subfolder, _, filelist in tqdm(sorted(os.walk(audio_dir))):
            for fname in filelist:
                audio_path = os.path.join(subfolder, fname)
                try:
                    n += 1
                    res = pool.apply_async(load_and_resample, (audio_path, n))
                    results.append(res)
                except:
                    continue
        pool.close()
        pool.join()
        for res in tqdm(results):
            get_res = res.get()
            if get_res is None:
                continue
            json_lines.append(get_res)
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
