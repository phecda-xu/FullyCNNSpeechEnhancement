"""
Summary:  Train, inference and evaluate speech enhancement.
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import numpy as np
import os
import pickle
# import cPickle
import h5py
import argparse
import time
import glob
import matplotlib.pyplot as plt

import prepare_data as pp_data
import config as cfg
from keras.models import load_model
from utils.utility import merge_magphase
import librosa
from utils.utility import divide_magphase
from pyaudio import PyAudio, paInt16
import wave
def demo(args):
    """Inference all test data, write out recovered wavs to disk.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      n_concat: int, number of frames to concatenta, should equal to n_concat
          in the training stage.
      iter: int, iteration of model to load.
      visualize: bool, plot enhanced spectrogram for debug.
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = args.n_concat
    iter = args.iteration

    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = True

    # Load model.
    model_path = os.path.join(workspace, "models", "%ddb" % int(tr_snr), "senn.h5")
    model = load_model(model_path)

    # Load test data.
    if args.online:
        print('recording....')
        recordfile ='record.wav'
        my_record(recordfile,16000, 2)
        print('recording end')
        (data,_) = pp_data.read_audio(recordfile,16000)
    else:
        testfile = 'test.wav'
        (data,_) = pp_data.read_audio(testfile,16000)
    mixed_complx_x = pp_data.calc_sp(data, mode='complex')
    mixed_x, mixed_phase = divide_magphase(mixed_complx_x, power=1)

    # Predict.
    pred = model.predict(mixed_x)
    # Recover enhanced wav.
    pred_sp = pred #np.exp(pred)
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    hop_size = n_window - n_overlap
    ham_win = np.sqrt(np.hanning(n_window))
    stft_reconstructed_clean = merge_magphase(pred_sp, mixed_phase)
    stft_reconstructed_clean =stft_reconstructed_clean.T
    signal_reconstructed_clean = librosa.istft(stft_reconstructed_clean, hop_length=hop_size,window=ham_win)
    signal_reconstructed_clean = signal_reconstructed_clean*32768
    s = signal_reconstructed_clean.astype('int16')

    # Write out enhanced wav.
    # out_path = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.wav" % na)
    # pp_data.create_folder(os.path.dirname(out_path))
    pp_data.write_audio('enhance.wav', s, fs)

def wavread(filename):
    fp=wave.open(filename,'rb')
    nf=fp.getnframes()#获取文件的采样点数量
    print('sampwidth:',fp.getsampwidth())
    print('framerate:',fp.getframerate())
    print('channels:',fp.getnchannels())
    f_len=nf*2#文件长度计算，每个采样2个字节
    audio_data=fp.readframes(nf)
    return audio_data

def save_wave_file(filename,data,channels,sampwidth,framerate):
    '''save the date to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)#声道
    wf.setsampwidth(sampwidth)#采样字节 1 or 2
    wf.setframerate(framerate)#采样频率 8000 or 16000
    wf.writeframes(b"".join(data))#https://stackoverflow.com/questions/32071536/typeerror-sequence-item-0-expected-str-instance-bytes-found
    wf.close()

def my_record(filename,framerate,seconds):
    pa=PyAudio()
    stream=pa.open(format = paInt16,channels=1,
                   rate=framerate,input=True,
                   frames_per_buffer=1600)
    my_buf=[]
    count=0
    while count<seconds*20:#控制录音时间
        string_audio_data = stream.read(1600)#一次性录音采样字节大小
        my_buf.append(string_audio_data)
        count+=1
        print('.')
    save_wave_file(filename,my_buf,1,2,16000)
    stream.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--online', type=int, required=True)
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    args = parser.parse_args()
    if args.mode == 'inference':
        demo(args)
    else:
        raise Exception("Error!")