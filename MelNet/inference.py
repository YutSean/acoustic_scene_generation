import os
import glob
import argparse
import torch
import torch.nn as nn
import audiosegment
import matplotlib.pyplot as plt
import numpy as np
import math
import random

from utils.plotting import plot_spectrogram_to_numpy
from utils.reconstruct import Reconstruct
from utils.constant import t_div
from utils.hparams import HParam
from model.model import MelNet
# --------------------------------------------------
from datasets.wavloader import create_dataloader
from multiprocessing import Pool
import pdb
# --------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_root = "/mnt/IDMT-WORKSPACE/DATA-STORE/xiaoyg/ma_xiaoyg-master/MelNet/generated_audios"


def store(generated, path, hp, idx, class_label):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(generated, os.path.join(path, '{}_{}.pt'.format(class_label, idx)))
    spectrogram = plot_spectrogram_to_numpy(generated[0].cpu().detach().numpy())
    plt.imsave(os.path.join(path, '{}_{}.png'.format(class_label, idx)), spectrogram.transpose((1, 2, 0)))
    with torch.enable_grad():
        waveform, wavespec = Reconstruct(hp).inverse(generated[0])
    wavespec = plot_spectrogram_to_numpy(wavespec.cpu().detach().numpy())
    plt.imsave(os.path.join(path, 'Final {}_{}.png'.format(class_label, idx)), wavespec.transpose((1, 2, 0)))

    waveform = waveform.unsqueeze(-1)
    waveform = waveform.cpu().detach().numpy()
    waveform *= 32768 / waveform.max()
    waveform = waveform.astype(np.int16)
    audio = audiosegment.from_numpy_array(
        waveform,
        framerate=hp.audio.sr
    )
    audio.export(os.path.join(path, '{}_{}.wav'.format(class_label, idx)), format='wav')

def get_gaussian_fileter(kernel_size, sigma, channels):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),

                                )

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--infer_config', type=str, required=True,
                        help="yaml file for inference configuration")
    parser.add_argument('-t', '--timestep', type=int, default=240,
                        help="timestep of mel-spectrogram to generate")
    parser.add_argument('-n', '--name', type=str, default="result", required=False,
                        help="Name for sample")
    parser.add_argument('-i', '--input', type=str, default=None, required=False,
                        help="Input for conditional generation, leave empty for unconditional")
    args = parser.parse_args()

    hp = HParam(args.config)
    infer_hp = HParam(args.infer_config)

    assert args.timestep % t_div[hp.model.tier] == 0, \
        "timestep should be divisible by %d, got %d" % (t_div[hp.model.tier], args.timestep)

    model = MelNet(hp, args, infer_hp).cuda()
    model.load_tiers()
    model.eval()

    args.tts = False
    args.tier = 1
    args.batch_size = 1

    testloader = create_dataloader(hp, args, train=False)

    gaussian_filter = get_gaussian_fileter(3, 1, 1)

    gaussian_filter = gaussian_filter.cuda()

    dependence_length = 62

    # with torch.no_grad():
    #     # generated = model.sample(args.input)
    #     for idx in range(1):
    #         source, target = testloader.dataset[idx]
    #         source = torch.from_numpy(source)
    #         data_ratio = 1
    #         args.timestep = source.size()[-1] - int(source.size()[-1] * data_ratio)
    #         if data_ratio == 0:
    #             generated = model.sample(condition=None)
    #         else:
    #             generated = model.sample(source[:, :int(source.size()[-1] * data_ratio)].unsqueeze(0).cuda())
    #         with torch.no_grad():
    #             generated = gaussian_filter(generated.unsqueeze(0)).squeeze(0)
    #
    #         # generated = source.unsqueeze(0).cuda()
    #         p = os.path.join(data_root, args.name)
    #         store(generated, p, hp, idx)

    class_labels = [
        'airport',
        'bus',
        'metro',
        'metro_station',
        'park',
        'public_square',
        'shopping_mall',
        'street_pedestrian',
        'street_traffic',
        'tram',
    ]

    label_count = {
        'airport': 0,
        'bus': 0,
        'metro': 0,
        'metro_station': 0,
        'park': 0,
        'public_square': 0,
        'shopping_mall': 10,
        'street_pedestrian': 0,
        'street_traffic': 0,
        'tram': 0
    }

    # store(mel[0], p, hp, 0, class_labels[1])
    with torch.no_grad():

        for idx in range(10, len(testloader.dataset)):
            source, target, label = testloader.dataset[-idx-1]
            label_count[class_labels[label]] += 1
            source = torch.from_numpy(source).unsqueeze(0)
            if label != 6:
                continue
            # for i in range(10):
            start_idx = random.randint(0, 626 - dependence_length)
            generated = model.sample_dependence(source[:, :, start_idx: start_idx + dependence_length].cuda(), label, dependence_length)
            generated = generated[:, :, -args.timestep:]
            generated = gaussian_filter(generated.unsqueeze(0))
            p = os.path.join(data_root, args.name)
            for g in generated:
                store(g, p, hp, label_count[class_labels[label]], class_labels[label])


