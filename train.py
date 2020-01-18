import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models import FastDVDnet, MultiFiberNet3d
from models.fastdvd_net import fastdvdnet_denoise_sequence

from dataloaders import DaliVideoLoader

from utils import add_video_gaussian_noise


def main(**args):
	""" Performs the main training loop
	"""
	start_time = time.time()
	# Load dataset
	print('> Loading datasets ...')

	train_loader = DaliVideoLoader(file_list = args['trainset_list'],
									batch_size = args['batch_size'],
									crop_size = args['patch_size'],
									sequence_length = args['temp_patch_size'],
									step = 1)

	# STEP 1: Model definition
	fastdvd_model = FastDVDnet().cuda()

	for seq_frames, labels in train_loader:  # CUDA torch.Tensor loader
		# shape of data: [N, T, C, H, W], e.g. T = 20
		# shape of label: [N, ]

		print("DATA DALI", seq_frames.shape, labels.shape, seq_frames.type())

		seqn_frames, noise_map = add_video_gaussian_noise(seq_frames, args['noise_ival'])
		print(seqn_frames.shape, noise_map.shape)

		denframes = fastdvdnet_denoise_sequence(seqn_frames, noise_map, fastdvd_model)

	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters
	parser.add_argument("--batch_size", type=int, default=1,
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=80,
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60],
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3,
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,
						help="Number of training steps to log psnr and perform orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=5,
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 25],
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25,
						help='noise level used on validation set')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=224, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=20, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=256000,
						help="Maximum number of patches")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs",
					 help='path of log files')
	parser.add_argument("--trainset_list", type=str, required=True,
					 help='path of trainset')
	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))