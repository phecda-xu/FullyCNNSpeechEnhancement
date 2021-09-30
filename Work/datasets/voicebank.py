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
from joblib import Parallel, delayed
from data_utils.utils import download, unpack

DATA_HOME = '~/data/SE'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target-dir",
    default=DATA_HOME + "/voicebank",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest-prefix",
    default="Work/voicebank/data/manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
parser.add_argument(
    "--sample-rate",
    default=16000,
    type=int,
    help="target audio sample rate")
args = parser.parse_args()


def load_and_resample(audio_path, n):
    path_dic = {"8000": "8K", "16000": "16K", "32000": "32K"}
    clean_audio_path = audio_path
    mix_audio_path = audio_path.replace("clean", "noisy")
    audio_data, samplerate = sf.read(clean_audio_path)
    mix_audio_data, samplerate = sf.read(mix_audio_path)
    if len(audio_data) < 100:
        return None
    if samplerate != args.sample_rate:
        audio_data = resampy.resample(
            audio_data, samplerate, args.sample_rate, filter='kaiser_best')
        mix_audio_data = resampy.resample(
            mix_audio_data, samplerate, args.sample_rate, filter='kaiser_best')
        samplerate = args.sample_rate
        clean_audio_path = clean_audio_path.replace("SE", "{}SE".format(path_dic[str(args.sample_rate)]))
        mix_audio_path = clean_audio_path.replace("clean", "noisy")
        try:
            if not os.path.exists(os.path.dirname(clean_audio_path)):
                os.makedirs(os.path.dirname(clean_audio_path))
            if not os.path.exists(os.path.dirname(mix_audio_path)):
                os.makedirs(os.path.dirname(mix_audio_path))
        except:
            print('Skip dir error...')
        sf.write(clean_audio_path, audio_data, samplerate)
        sf.write(mix_audio_path, mix_audio_data, samplerate)
    duration = float(len(audio_data) / samplerate)
    json_str = json.dumps(
        {
            'clean_audio_filepath': clean_audio_path,
            'mix_audio_filepath': mix_audio_path,
            'duration': duration,
        },
        ensure_ascii=False)
    return json_str


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    if not os.path.exists(os.path.dirname(manifest_path_prefix)):
        os.makedirs(os.path.dirname(manifest_path_prefix))
    data_sets = ['clean_trainset', 'clean_testset']
    for data_set in data_sets:
        json_lines = []
        n = 0
        audio_dir = os.path.join(data_dir, data_set)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            results = Parallel(n_jobs=-1)(
                delayed(load_and_resample)(os.path.join(subfolder, fname), n) for fname in filelist
            )
            json_lines.extend(results)
        manifest_path = manifest_path_prefix + '.' + data_set.split("_")[-1]
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')


def prepare_dataset(url_list, target_dir, manifest_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'dataset')
    if not os.path.exists(data_dir):
        for url in url_list:
            filaname = url.split('/')[-1]
            data_dir_sub = os.path.join(data_dir, '_'.join(filaname.split("_")[:2]))
            if not os.path.exists(data_dir_sub):
                os.makedirs(data_dir_sub)
            filepath = download(" --no-check-certificate " + url, target_dir)
            unpack(filepath, data_dir_sub)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    create_manifest(data_dir, manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)
    urls = ['https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip',
            'https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip',
            'https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip',
            'https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip']

    prepare_dataset(
        url_list=urls,
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix)


if __name__ == '__main__':
    main()
