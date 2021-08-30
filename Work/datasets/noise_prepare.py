# coding:utf-8
#
# Date : 2020.04.01
# Author: phecda-xu < >
#
# DEC:
#     noise manifest

import io
import os
import json
import resampy
import argparse
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Process, Queue, Pool, Manager


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target-dir",
    default="~/Noise/_Farfiled_background_",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest-prefix",
    default="Work/noise/manifest.farfiled.background",
    type=str,
    help="Filepath for output manifests. (default: %(default)s)")
parser.add_argument(
    "--sample-rate",
    default=16000,
    type=int,
    help="target audio sample rate")
args = parser.parse_args()


def load_and_resample(audio_path, n):
    path_dic = {"8000": "8K", "16000": "16K", "32000": "32K"}
    audio_data, samplerate = sf.read(audio_path)
    if len(audio_data) < 100:
        return None
    if samplerate != args.sample_rate:
        audio_data = resampy.resample(
            audio_data, samplerate, args.sample_rate, filter='kaiser_best')
        samplerate = args.sample_rate
        audio_path = audio_path.replace("Noise", "{}Noise".format(path_dic[str(args.sample_rate)]))
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
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path_prefix)
    data_types = ['train', 'dev', 'test']
    for data_type in data_types:
        json_lines = []
        pool = Pool()
        results = []
        n = 0
        audio_dir = os.path.join(data_dir, data_type)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for filename in tqdm(filelist):
                if filename.endswith('.wav'):
                    audio_path = os.path.join(data_dir, subfolder, filename)
                    n += 1
                    res = pool.apply_async(load_and_resample, (audio_path, n))
                    results.append(res)
        pool.close()
        pool.join()
        for res in results:
            get_res = res.get()
            if get_res is None:
                continue
            json_lines.append(get_res)
        manifest_path = manifest_path_prefix + '.' + data_type
        if not os.path.exists(os.path.dirname(manifest_path)):
            os.makedirs(os.path.dirname(manifest_path))
        with io.open(manifest_path, mode='w') as out_file:
            for line in json_lines:
                out_file.write(line + '\n')


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)
    create_manifest(args.target_dir, args.manifest_prefix)


if __name__ == '__main__':
    main()
