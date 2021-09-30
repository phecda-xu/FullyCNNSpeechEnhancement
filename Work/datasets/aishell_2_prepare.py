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
from data_utils.utils import unpack
from joblib import Parallel, delayed

DATA_HOME = '~/data/ASR'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target-dir",
    default=DATA_HOME + "/AISHELL-2",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest-prefix",
    default="Work/aishell_2/data/manifest",
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
    return json_str


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    if not os.path.exists(os.path.dirname(manifest_path_prefix)):
        os.makedirs(os.path.dirname(manifest_path_prefix))
    # split train/test/dev with speaker info
    speak_info_path = os.path.join(data_dir, 'iOS/data/', 'spk_info.txt')
    gender_list_male = []
    gender_list_female = []
    for line in codecs.open(speak_info_path, 'r', 'utf-8'):
        line = line.strip()
        if line == '':
            continue
        spk_id = line.split('\t')[0].strip()
        gender = line.split('\t')[2].strip()
        if gender == '女':
            gender_list_female.append(spk_id)
        else:
            gender_list_male.append(spk_id)

    gender_list_female.sort(key=lambda x: int(x[1:]))
    gender_list_male.sort(key=lambda x: int(x[1:]))

    # find all wav in each dataset and generate manifest file
    train_spk_list = gender_list_female[20:]
    train_spk_list.extend(gender_list_male[20:])
    test_spk_list = gender_list_female[10:20]
    test_spk_list.extend(gender_list_male[10:20])
    dev_spk_list = gender_list_female[0:10]
    dev_spk_list.extend(gender_list_male[0:10])

    train_json_lines = []
    test_json_lines = []
    dev_json_lines = []
    audio_dir = os.path.join(data_dir, 'iOS/data/wav')
    n = 0
    for subfolder, _, filelist in sorted(os.walk(audio_dir)):
        spk_id = os.path.basename(subfolder)
        results = Parallel(n_jobs=-1)(
            delayed(load_and_resample)(os.path.join(subfolder, fname), n) for fname in filelist
        )
        if spk_id in dev_spk_list:
            dev_json_lines.extend(results)
        elif spk_id in test_spk_list:
            test_json_lines.extend(results)
        else:
            train_json_lines.extend(results)
    # save manifest
    manifest_path = manifest_path_prefix + '.' + 'train'
    with codecs.open(manifest_path, 'w', 'utf-8') as fout:
        for line in train_json_lines:
            fout.write(line + '\n')

    manifest_path = manifest_path_prefix + '.' + 'test'
    with codecs.open(manifest_path, 'w', 'utf-8') as fout:
        for line in test_json_lines:
            fout.write(line + '\n')

    manifest_path = manifest_path_prefix + '.' + 'dev'
    with codecs.open(manifest_path, 'w', 'utf-8') as fout:
        for line in dev_json_lines:
            fout.write(line + '\n')

    manifest_path = manifest_path_prefix + '.' + 'all'
    with codecs.open(manifest_path, 'w', 'utf-8') as fout:
        for line in dev_json_lines:
            fout.write(line + '\n')
        for line in test_json_lines:
            fout.write(line + '\n')
        for line in train_json_lines:
            fout.write(line + '\n')


def prepare_dataset(target_dir, manifest_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir)
    if not os.path.exists(data_dir):
        filepath = os.path.join(target_dir, 'AISHELL-2.tar.gz')
        unpack(filepath, target_dir)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    create_manifest(data_dir, manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix)


if __name__ == '__main__':
    main()
