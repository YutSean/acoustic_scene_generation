import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from .gmm import sample_gmm
import pdb

# -----------------------------------------
from utils.plotting import plot_spectrogram_to_numpy
from utils.reconstruct import Reconstruct
from utils.constant import t_div
from utils.hparams import HParam
from model.model import MelNet
import matplotlib.pyplot as plt
import numpy as np
import audiosegment
import os
# -----------------------------------------
store_path_root = '/mnt/IDMT-WORKSPACE/DATA-STORE/xiaoyg/ma_xiaoyg-master/MelNet/generated_audios'

def validate(args, model, melgen, tierutil, testloader, criterion, writer, step, hp=None):
    model.eval()
    # torch.backends.cudnn.benchmark = False
    test_loss = []
    loader = tqdm(testloader, desc='Testing is in progress', dynamic_ncols=True)
    idx = 0
    with torch.no_grad():
        for input_tuple in loader:
            if args.tts:
                seq, text_lengths, source, target, audio_lengths = input_tuple
                mu, std, pi, alignment = model(
                    source.cuda(non_blocking=True),
                    seq.cuda(non_blocking=True),
                    text_lengths.cuda(non_blocking=True),
                    audio_lengths.cuda(non_blocking=True)
                )
            else:
                source, target, labels, audio_lengths = input_tuple
                mu, std, pi = model(
                    source.cuda(non_blocking=True),
                    audio_lengths.cuda(non_blocking=True),
                    labels.cuda(non_blocking=True)
                )
            # helper(sample_gmm(mu, std, pi).cuda(), 'generated_{}'.format(idx), hp, os.path.join(store_path_root, 'generated'))
            idx += 1
            loss = criterion(
                target.cuda(non_blocking=True),
                mu, std, pi,
                audio_lengths.cuda(non_blocking=True)
            )
            test_loss.append(loss)

        test_loss = sum(test_loss) / len(test_loss)
        audio_length = audio_lengths[0].item()
        # ts = source.cuda()
        # tt = target.cuda()
        # tr = sample_gmm(mu, std, pi).cuda()

        source = source[0].cpu().detach().numpy()[:, :audio_length]
        target = target[0].cpu().detach().numpy()[:, :audio_length]
        result = sample_gmm(mu[0], std[0], pi[0]).cpu().detach().numpy()[:, :audio_length]
        if args.tts:
            alignment = alignment[0].cpu().detach().numpy()[:, :audio_length]
        else:
            alignment = None
        writer.log_validation(test_loss, source, target, result, alignment, step)
    # helper(tr, 'result_{}'.format(step), hp, os.path.join(store_path_root, 'result'))
    # helper(ts, 'source_{}'.format(step), hp, os.path.join(store_path_root, 'source'))
    # helper(tt, 'target_{}'.format(step), hp, os.path.join(store_path_root, 'target'))
    model.train()
    # torch.backends.cudnn.benchmark = True

def helper(data, name, hp, store_path):

    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)
    spectrogram = plot_spectrogram_to_numpy(data[0].cpu().detach().numpy())
    plt.imsave(os.path.join(store_path, name + '.png'), spectrogram.transpose((1, 2, 0)))
    with torch.enable_grad():
        waveform, wavespec = Reconstruct(hp).inverse(data[0], iters=2000)
    wavespec = plot_spectrogram_to_numpy(wavespec.cpu().detach().numpy())
    plt.imsave(os.path.join(store_path, 'Final ' + name + '.png'), wavespec.transpose((1, 2, 0)))

    waveform = waveform.unsqueeze(-1)
    waveform = waveform.cpu().detach().numpy()
    waveform *= 32768 / waveform.max()
    waveform = waveform.astype(np.int16)
    audio = audiosegment.from_numpy_array(
        waveform,
        framerate=hp.audio.sr
    )
    audio.export(os.path.join(store_path, name + '.wav'), format='wav')