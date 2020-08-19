import os
import torch
from torch import nn, optim
import datetime as d
import re
import json

from wavenet.audiodata import AudioData, AudioLoader
from wavenet.models import Model, Generator
from wavenet.utils import list_files

from argparse import ArgumentParser
from multiprocessing import Pool, TimeoutError

import pdb


def set_args():
    parser = ArgumentParser(description='Wavenet demo')
    parser.add_argument('--data', type=str, default='your data path',
                        help='folder to training set of .wav files')
    parser.add_argument('--x_len', type=int, default=5122, help='length of input')
    parser.add_argument('--num_classes', type=int, default=256, help='number of discrete output levels')
    parser.add_argument('--num_layers', type=int, default=10, help='number of convolutional layers per block')
    parser.add_argument('--num_blocks', type=int, default=5, help='number of repeating convolutional layer blocks')
    parser.add_argument('--dilation_channels', type=int, default=32, help='number of neurons per layer')
    parser.add_argument('--residual_channels', type=int, default=32, help='number of neurons per layer')
    parser.add_argument('--skip_channels', type=int, default=32, help='number of neurons per layer')
    parser.add_argument('--end_channels', type=int, default=32, help='number of neurons per layer')
    parser.add_argument('--kernel_size', type=int, default=2, help='width of convolutional kernel')
    parser.add_argument('--learn_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--output_width', type=int, default=1, help='learning rate')
    parser.add_argument('--bias', type=bool, default=True, help='learning rate')
    parser.add_argument('--step_size', type=int, default=100, help='step size of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--disp_interval', type=int, default=1000, help='number of epochs in between display messages')
    parser.add_argument('--model_file', type=str, default='your dir to store model file', help='filename of model')
    parser.add_argument('--visdom', type=bool, default=False, help='flag to track variables in visdom')
    parser.add_argument('--new_seq_len', type=int, default=5000, help='length of sequence to predict')
    parser.add_argument('--device', type=str, default='default', help='device to use')
    parser.add_argument('--resume_train', type=bool, default=False, help='whether to resume training existing models')
    parser.add_argument('--root_dir', type=str, default='your dir path to store the configuration.')
    parser.add_argument('--train_steps', type=int, default=100000, help='number of training steps.')

    args = parser.parse_args()
    # args.class_labels = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square', 'street_traffic',
    #                 'tram', 'bus', 'metro', 'park']
    args.class_labels = ['music']
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
    model = torch.load(os.path.join(args.model_file,  'model_{}.pt'.format(num)))
    return model, num


def load_dataset(t):
    idx = t[0]
    label = t[1]
    filelist = list_files(os.path.join(args.data, label))
    temp = AudioData(filelist, wave_model.receptive_field, y_len=args.output_width,
                     num_classes=args.num_classes, store_tracks=False, class_label=idx)
    return temp


if __name__ == '__main__':
    args = load_parameters()

    # construct model
    wave_model = Model(
                       layers=args.num_layers,
                       num_classes=args.num_classes,
                       blocks=args.num_blocks,
                       kernel_size=args.kernel_size,
                       dilation_channels=args.dilation_channels,
                       residual_channels=args.residual_channels,
                       skip_channels=args.skip_channels,
                       end_channels=args.end_channels,
                       bias=args.bias,
                       output_length=args.output_width,
                       )
    if torch.cuda.is_available():
        args.device = 'cuda'

    if not (args.device == 'default'):
        wave_model.set_device(torch.device(args.device))

    # create dataset and dataloader
    datasets = []
    meta = []
    for idx, label in enumerate(args.class_labels):
        meta.append((idx, label))
    # for idx, label in enumerate(args.class_labels):
    #     filelist = list_files(os.path.join(args.data, label))
    #     temp = AudioData(filelist, wave_model.receptive_field, y_len=args.output_width,
    #                      num_classes=args.num_classes, store_tracks=False, class_label=idx)
    #     datasets.append(temp)
    with Pool(processes=10) as pool:
        datasets = pool.map(load_dataset, meta)
    # datasets.append(load_dataset(meta[0]))
    dataset = torch.utils.data.ConcatDataset(datasets)
    pivot = int(len(dataset) * 0.7)
    # train_set = torch.utils.data.Subset(dataset, range(pivot + 1))
    # train_set = torch.utils.data.Subset(dataset, range(3000))
    train_set = dataset
    valid_set = torch.utils.data.Subset(dataset, range(pivot + 1, len(dataset)))

    # manage small datasets with big batch sizes
    if args.batch_size > len(dataset):
        args.batch_size = len(dataset)
    dataloader = AudioLoader(train_set, batch_size=args.batch_size,
                             num_workers=args.num_workers)
    valid_loader = AudioLoader(valid_set, batch_size=args.batch_size,
                               num_workers=args.num_workers)

    # load/train model
    start_epoch = 0

    if args.resume_train:
        print('Resuming training.')

        if os.path.exists(args.model_file):
            print('Loading model data from file: {}'.format(args.model_file))
            temp, resume_epoch = load_model(args.model_file)
            if temp:
                wave_model = temp
                start_epoch = resume_epoch
            else:
                print('Model data not found: {}'.format(args.model_file))
                print('Training new model')
    else:
        print('Model data not found: {}'.format(args.model_file))
        print('Training new model.')
        args.resume_train = True

    if args.resume_train:
        wave_model.criterion = nn.CrossEntropyLoss()
        wave_model.optimizer = optim.Adam(wave_model.parameters(), 
                                          lr=args.learn_rate)
        wave_model.scheduler = optim.lr_scheduler.StepLR(wave_model.optimizer, 
                                                         step_size=args.step_size, 
                                                         gamma=args.gamma)

        try:
            print('Training starts at {}'.format(d.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")))
            wave_model.train(dataloader,
                             num_epochs=args.num_epochs,
                             disp_interval=args.disp_interval,
                             use_visdom=args.visdom,
                             model_dir=args.model_file,
                             start_epoch=start_epoch,
                             num_steps=args.train_steps)
            print('Training finished at {}'.format(d.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")))
        except KeyboardInterrupt:
            print('Training stopped!')
            print('Training stopped at {}'.format(d.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")))



        # print('Saving model data to file: {}'.format(args.model_file))
        # torch.save(wave_model.state_dict(), args.model_file)

    # predict sequence with model
    # wave_generator = Generator(wave_model, dataset)
    # seed = dataset.tracks[0]['audio'][:args.x_len]
    # y = wave_generator.run(seed, args.new_seq_len, disp_interval=100)
    # y = wave_generator.run(torch.Tensor([0] * args.x_len), args.new_seq_len, disp_interval=100)
    # dataset.save_wav('./tmp.wav', y, dataloader.dataset.sample_rate)
