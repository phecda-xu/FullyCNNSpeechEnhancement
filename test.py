import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import prepare_data as pp_data
import config as cfg
from keras.models import load_model
from utils.utility import merge_magphase
import librosa
from utils.utility import divide_magphase


def test(args):
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
	model_path = os.path.join(workspace, "models", "%ddb" % int(tr_snr), "FullyCNN.h5")
	model = load_model(model_path)


	# Load test data.
	feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "%ddb" % int(te_snr))
	names = os.listdir(feat_dir)

	for (cnt, na) in enumerate(names):
		# Load feature.
		feat_path = os.path.join(feat_dir, na)
		data = pickle.load(open(feat_path, 'rb'))
		[mixed_cmplx_x, speech_cmplx_x, noise_x, alpha, na] = data

		mixed_x, mixed_phase = divide_magphase(mixed_cmplx_x, power=1)  # power=1 为幅度谱
		speech_x, clean_phase = divide_magphase(speech_cmplx_x, power=1)

		# Predict.
		pred = model.predict(mixed_x)
		print(cnt, na)
		# Debug plot.
		if args.visualize:
			fig, axs = plt.subplots(3, 1)
			axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
			axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
			axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
			axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
			axs[1].set_title("Clean speech log spectrogram")
			axs[2].set_title("Enhanced speech log spectrogram")
			for j1 in range(3):
				axs[j1].xaxis.tick_bottom()
			plt.tight_layout()
			plt.show()

		# Recover enhanced wav.
		pred_sp = pred  # np.exp(pred)
		n_window = cfg.n_window
		n_overlap = cfg.n_overlap
		hop_size = n_window - n_overlap
		ham_win = np.sqrt(np.hanning(n_window))
		stft_reconstructed_clean = merge_magphase(pred_sp, mixed_phase)
		stft_reconstructed_clean = stft_reconstructed_clean.T
		signal_reconstructed_clean = librosa.istft(stft_reconstructed_clean, hop_length=hop_size,window=ham_win)
		signal_reconstructed_clean = signal_reconstructed_clean*32768
		s = signal_reconstructed_clean.astype('int16')

		# Write out enhanced wav.
		out_path = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.wav" % na)
		pp_data.create_folder(os.path.dirname(out_path))
		pp_data.write_audio(out_path, s, fs)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='mode')
	parser_inference = subparsers.add_parser('inference')
	parser_inference.add_argument('--workspace', type=str, default='workspace')
	parser_inference.add_argument('--tr_snr', type=float, default=0)
	parser_inference.add_argument('--te_snr', type=float, default=0)
	parser_inference.add_argument('--n_concat', type=int, default=7)
	parser_inference.add_argument('--iteration', type=int, default=100)
	parser_inference.add_argument('--visualize', action='store_true', default=False)
	args = parser.parse_args(['inference'])
	if args.mode == 'inference':
		test(args)
	else:
		raise Exception("Error!")
