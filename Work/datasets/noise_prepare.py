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
import argparse
import soundfile
from tqdm import tqdm


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default="~/_Farfiled_background_",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="Work/noise/manifest.farfiled.background",
    type=str,
    help="Filepath for output manifests. (default: %(default)s)")
args = parser.parse_args()


def create_manifest(data_dir, manifest_path_prefix):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    data_types = ['train', 'dev', 'test']
    for data_type in data_types:
        del json_lines[:]
        audio_dir = os.path.join(data_dir, data_type)
        for subfolder, _, filelist in tqdm(sorted(os.walk(audio_dir))):
            for filename in filelist:
                if filename.endswith('.wav'):
                    filepath = os.path.join(data_dir, subfolder, filename)
                    audio_data, samplerate = soundfile.read(filepath)
                    duration = float(len(audio_data)) / samplerate
                    json_lines.append(
                        json.dumps({
                            'audio_filepath': filepath,
                            'duration': duration,
                        }))
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
