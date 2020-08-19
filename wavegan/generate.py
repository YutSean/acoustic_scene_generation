import argparse
import logging
import os
import json
import datetime
import torch
import pprint
import pickle as pk
from sample import get_all_audio_filepaths, create_data_split
from wavegan import WaveGANDiscriminator, WaveGANGenerator
from wgan import train_wgan
from log import init_console_logger
from utils import save_generated
import pdb

def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a WaveGAN on a given set of audio')

    parser.add_argument('-ms', '--model-size', dest='model_size', type=int, default=64, help='Model size parameter used in WaveGAN')
    parser.add_argument('-pssf', '--phase-shuffle-shift-factor', dest='shift_factor', type=int, default=2, help='Maximum shift used by phase shuffle')
    parser.add_argument('-psb', '--phase-shuffle-batchwise', dest='batch_shuffle', action='store_true', help='If true, apply phase shuffle to entire batches rather than individual samples')
    parser.add_argument('-ppfl', '--post-proc-filt-len', dest='post_proc_filt_len', type=int, default=512, help='Length of post processing filter used by generator. Set to 0 to disable.')
    parser.add_argument('-lra', '--lrelu-alpha', dest='alpha', type=float, default=0.2, help='Slope of negative part of LReLU used by discriminator')
    parser.add_argument('-vr', '--valid-ratio', dest='valid_ratio', type=float, default=0.1, help='Ratio of audio files used for validation')
    parser.add_argument('-tr', '--test-ratio', dest='test_ratio', type=float, default=0.1, help='Ratio of audio files used for testing')
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=16, help='Batch size used for training')
    parser.add_argument('-ne', '--num-epochs', dest='num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-bpe', '--batches-per-epoch', dest='batches_per_epoch', type=int, default=10, help='Batches per training epoch')
    parser.add_argument('-ng', '--ngpus', dest='ngpus', type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('-du', '--discriminator-updates', dest='discriminator_updates', type=int, default=5, help='Number of discriminator updates per training iteration')
    parser.add_argument('-ld', '--latent-dim', dest='latent_dim', type=int, default=300, help='Size of latent dimension used by generator')
    parser.add_argument('-eps', '--epochs-per-sample', dest='epochs_per_sample', type=int, default=1, help='How many epochs between every set of samples generated for inspection')
    parser.add_argument('-ss', '--sample-size', dest='sample_size', type=int, default=20, help='Number of inspection samples generated')
    parser.add_argument('-rf', '--regularization-factor', dest='lmbda', type=float, default=10.0, help='Gradient penalty regularization factor')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=1e-4, help='Initial ADAM learning rate')
    parser.add_argument('-bo', '--beta-one', dest='beta1', type=float, default=0.5, help='beta_1 ADAM parameter')
    parser.add_argument('-bt', '--beta-two', dest='beta2', type=float, default=0.9, help='beta_2 ADAM parameter')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('--audio_dir', type=str,
                        default="/mnt/IDMT-WORKSPACE/DATA-STORE/xiaoyg/environment_audio",
                        help='Path to directory containing audio files')
    parser.add_argument('--output_dir', type=str,
                        default="/mnt/IDMT-WORKSPACE/DATA-STORE/xiaoyg/ma_xiaoyg-master/WaveGAN/models",
                        help='Path to directory where model files will be output')
    parser.add_argument('-c', '--class-num', dest='class_num', type=int, default=10, help='Number of classes.')
    args = parser.parse_args()
    return vars(args)


exp = '20200702222625'
class_labels = ['airport', 'bus', 'metro', 'metro_station', 'park',
                'public_square', 'shopping_mall', 'street_pedestrian',
                'street_traffic', 'tram']

if __name__ == '__main__':
    args = parse_arguments()
    epoch = 100
    model_dir = os.path.join(args['output_dir'], exp)
    model_gen = torch.load(os.path.join(model_dir, 'model_gen_{}.pt').format(epoch))
    sample_size = 100
    latent_dim = args['latent_dim']
    start_idx = 0
    for _ in range(10):
        for i in range(10):
            label = torch.tensor([i] * sample_size, dtype=torch.long).cuda()
            output_path = os.path.join(args['output_dir'], 'generated_audio')
            output_path = os.path.join(output_path, class_labels[i])
            sample_noise = torch.Tensor(sample_size, latent_dim).uniform_(-1, 1).cuda()
            with torch.no_grad():
                generated = model_gen(sample_noise, label).cpu()
            save_generated(generated.numpy(), output_path, start_idx=start_idx)
        start_idx += sample_size

