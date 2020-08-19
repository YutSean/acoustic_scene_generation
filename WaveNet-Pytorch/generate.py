import torch
import os
import re
from wavenet.audiodata import AudioData, AudioLoader
from wavenet.models import Model, Generator
from wavenet.utils import list_files
from argparse import ArgumentParser
from multiprocessing import Pool, TimeoutError
import multiprocessing as mp
import numpy as np
import json
import librosa
import soundfile as sf

import pdb

mp.set_start_method("spawn", force=True)

def set_args():
    parser = ArgumentParser(description='Wavenet demo')
    parser.add_argument('--data', type=str, default='/mnt/IDMT-WORKSPACE/DATA-STORE/xiaoyg/environment_audio/airport', help='folder to training set of .wav files')
    parser.add_argument('--x_len', type=int, default=2**14, help='length of input')
    parser.add_argument('--num_classes', type=int, default=256, help='number of discrete output levels')
    parser.add_argument('--num_layers', type=int, default=13, help='number of convolutional layers per block')
    parser.add_argument('--num_blocks', type=int, default=2, help='number of repeating convolutional layer blocks')
    parser.add_argument('--num_hidden', type=int, default=32, help='number of neurons per layer')
    parser.add_argument('--kernel_size', type=int, default=2, help='width of convolutional kernel')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--step_size', type=int, default=50, help='step size of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=5, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--disp_interval', type=int, default=1, help='number of epochs in between display messages')
    parser.add_argument('--model_file', type=str, default='/mnt/IDMT-WORKSPACE/DATA-STORE/xiaoyg/wavenet_models_airport/lr=0.0001/', help='filename of model')
    parser.add_argument('--visdom', type=bool, default=False, help='flag to track variables in visdom')
    parser.add_argument('--new_seq_len', type=int, default=10, help='length of sequence to predict')
    parser.add_argument('--device', type=str, default='default', help='device to use')
    parser.add_argument('--resume_train', type=bool, default=False, help='whether to resume training existing models')
    parser.add_argument('--root_dir', type=str, default='/mnt/IDMT-WORKSPACE/DATA-STORE/xiaoyg/wavenets/test_wavenet')

    args = parser.parse_args()
    return args


def load_parameters():
    args = set_args()
    parameter_path = os.path.join(args.root_dir, 'hyperparameters.json')
    args.model_file = os.path.join(args.root_dir, 'models')
    args.generated_sounds = os.path.join(args.root_dir, 'generated_sounds')
    if os.path.exists(parameter_path):
        with open(parameter_path) as js_file:
            parameters = json.load(js_file)
        for key, val in parameters.items():
            setattr(args, str(key), val)
    else:
        os.makedirs(args.root_dir, exist_ok=True)
        with open(parameter_path, mode='w') as js_file:
            json.dump(args.__dict__, fp=js_file, indent=1)
    return args


def load_model(path, epoch=0):
    args = load_parameters()
    num = 0
    if not os.path.exists(path):
        raise ValueError('No such directory: ' + path)
    else:
        models = os.listdir(path)
        if not epoch:
            for model_name in models:
                num = max(int(re.findall(r'\d+', model_name)[0]), num)
        else:
            num = epoch
    model = torch.load(os.path.join(args.model_file, 'model_{}.pt'.format(num)))
    return model

def load_dataset(t):
    idx, label, receptive_field, args = t
    filelist = list_files(os.path.join(args.data, label))[:10]
    temp = AudioData(filelist, receptive_field, y_len=args.output_width,
                     num_classes=args.num_classes, store_tracks=False, class_label=idx)
    return temp

def generate_run(param):
    idx = param[0]
    class_label = param[1]
    dataset = param[2]
    wave_model = param[3]
    args = param[4]
    wave_generator = Generator(wave_model, dataset)
    start_idx = 0
    seeds = []
    for i, track in enumerate(dataset.tracks[start_idx: start_idx + 5]):
        print('Start generate sample {} in class {}'.format(i + 1, class_label))
        start = 0
        for start in range(0, 32000, wave_model.receptive_field):
            seed = track['audio'][start: start + wave_model.receptive_field]
            seed = torch.from_numpy(seed).to(wave_model.device)
            seeds.append(seed)
    seed = torch.stack(seeds)
    y = wave_generator.run(seed, args.new_seq_len, disp_interval=100, label=idx)
    for i, sample in enumerate(y):
        # dataset.save_wav(os.path.join(args.generated_sounds, '{}_{}.wav'.format(class_label, i + start_idx)), temp, dataset.sample_rate)
        sample.resize(sample.size, 1)
        target = os.path.join(args.generated_sounds, '{}_{}.wav'.format(class_label, i + start_idx))
        sf.write(target, sample, dataset.sample_rate, 'PCM_16')

def generate():
    args = load_parameters()
    wave_model = load_model(args.model_file)
    if not os.path.exists(args.generated_sounds):
        os.makedirs(args.generated_sounds, exist_ok=True)

    datasets = []
    meta = []
    for idx, label in enumerate(args.class_labels):
        meta.append((idx, label, wave_model.receptive_field, args))
    # for idx, label in enumerate(args.class_labels):
    #     filelist = list_files(os.path.join(args.data, label))
    #     temp = AudioData(filelist, wave_model.receptive_field, y_len=args.output_width,
    #                      num_classes=args.num_classes, store_tracks=False, class_label=idx)
    #     datasets.append(temp)
    with Pool(processes=5) as pool:
        datasets = pool.map(load_dataset, meta)

    params = []
    for idx, class_label in enumerate(args.class_labels):
        params.append((idx, class_label, datasets[idx], wave_model, args))
    generate_run(params[0])
    # with Pool(processes=5) as pool:
    #     output = pool.map(generate_run, params)


if __name__ == '__main__':
    print(mp.get_start_method())
    generate()
