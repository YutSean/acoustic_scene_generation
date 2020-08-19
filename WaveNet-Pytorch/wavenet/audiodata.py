# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:25:25 2017

@author: bricklayer
"""

import numpy as np

import scipy.signal
import soundfile as sf
import librosa
import os
import h5py
import pdb

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

from .muencoder import MuEncoder


class AudioData(Dataset):
    def __init__(self, track_list, x_len, y_len=1, bitrate=16, twos_comp=True, 
                 num_classes=256, store_tracks=False, encoder=None, is_scalar=False, sample_step=2000, gc=None, class_label=None):
        self.data = []
        self.tracks = []
        self.receptive_field = x_len
        self.y_len = y_len
        self.num_channels = 1
        self.num_classes = num_classes
        self.bitrate = bitrate
        self.datarange = (-2**(bitrate - 1), 2**(bitrate - 1) - 1)
        self.twos_comp = twos_comp
        self.is_scalar = is_scalar
        # self.bins = np.linspace(-1, 1, num_classes)
        self.data_len = 0
        self.sample_step = sample_step
        self.tracks_buckets = {}
        self.class_label = class_label
        self.gc = gc

        if encoder is None:
            self.encoder = MuEncoder(self.datarange)
        if not track_list:
            raise FileNotFoundError('The data directory contains no file with postfix .wav')
        self.data_root = os.path.dirname(track_list[0])


        # data_root = os.path.join(os.path.dirname(track_list[0]), "data.h5")
        # if os.path.exists(data_root):
        #     hf = h5py.File(data_root, 'r')
        #     matrix = hf.get('dataset')
        #     self.data_array = np.array(matrix)
        #     hf.close()
        # else:
        #     data_matrix = None
        self.track_list = track_list
        store_tracks = True
        for idx, track in enumerate(track_list):
            audio, dtype, sample_rate = self._load_audio_from_wav(track)
            audio = self.trim_silence(audio)
            audio = audio.reshape(-1, 1)
            audio = librosa.mu_compress(audio) + 128
            # if len(audio) > 160000:
            #     audio = audio[:160000]
            # elif len(audio) < 160000:
            #     audio = np.pad(audio, (160000 - len(audio), 0), mode='constant', constant_values=(0, 0))
            # if data_matrix is None:
            #     data_matrix = audio
            # else:
            #     data_matrix = np.concatenate([data_matrix, audio], axis=0)

        # audio = np.pad(audio, ((self.receptive_field, 0), (0, 0)), mode='constant', constant_values=0)

            if store_tracks:
                self.tracks.append({'name': track,
                                    'audio': audio,
                                    'sample_rate': sample_rate})
        # This is a problem this dataset is none-overlapping sampling,
        # that means larger x_len results in smaller data volume.
        #     for i in range(0, len(audio) - self.receptive_field - y_len, self.sample_step):
            for i in range(0, len(audio) - self.receptive_field - y_len, self.receptive_field):
                # x, y = self._extract_segment(audio, self.receptive_field, y_len, start_idx=i)
                self.data.append({'file_idx': idx,
                                  'start_idx': i,
                                  })
            #     x, y = self.preprocess(x, y)
            #     self.data.append({'x': x, 'y': y})
                # self.data.append((track, i))
            # start_idx = self.data_len
            # self.data_len += (len(audio) - x_len - y_len) // self.sample_step
            # self.tracks_buckets[track] = (start_idx, self.data_len - 1)
            # self.data.append(audio)
            # self.data_array = data_matrix
            # hf = h5py.File(data_root, 'w')
            # hf.create_dataset('dataset', data=self.data_array)
            # hf.close()

        self.dtype = np.dtype('int16')
        self.sample_rate = 16000
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # track, start_i = self.data[idx]
        # audio, dtype, sample_rate = self._load_audio_from_wav(track)
        # x, y = self._extract_segment(audio, self.x_len, self.y_len, start_idx=start_i)
        # x, y = self.preprocess(x, y)
        # return self._to_tensor(x, y)
        # return self._to_tensor(self.data[idx]['x'], self.data[idx]['y'])
        # self.encoder.encode(self.data[idx])
        # x = self.data[idx]['x']
        # y = self.data[idx]['y']
        file_idx = self.data[idx]['file_idx']
        start_idx = self.data[idx]['start_idx']
        audio = self.tracks[file_idx]['audio']

        x, y = self._extract_segment(audio, self.receptive_field, self.y_len, start_idx=start_idx)
        if self.class_label is None:
            return x, y
        else:
            return x, y, self.class_label

    def _load_audio_from_wav(self, filename):
        # read audio
        sample_rate = librosa.core.get_samplerate(filename)
        # with sf.SoundFile(filename) as myfile:
        #     audio = myfile.read(frames=-1, dtype='float32')
        #     sample_rate = myfile.samplerate
        if sample_rate > 16000:
            sample_rate = 16000
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = librosa.util.normalize(audio)

        # pdb.set_trace()
        # assert(audio.dtype == 'int16') # assume audio is int16 for now
        dtype = audio.dtype

        # combine channels
        audio = np.array(audio)
        # audio = audio.reshape(-1, 1)
        audio = np.interp(audio, (audio.min(), audio.max()), (-1, +1))

        # if len(audio.shape) > 1:
        #     audio = np.mean(audio, 1)
        # if sample_rate > 16000:
        #     resample_factor = sample_rate // 16000
        #     audio = scipy.signal.decimate(audio, resample_factor)
        #     sample_rate = 16000
        # audio = np.around(audio).astype(dtype)
        return audio, dtype, sample_rate
    
    def _extract_segment(self, audio, x_len, y_len, start_idx=None):
        num_samples = audio.shape[0]
        num_points = x_len + y_len
        
        if start_idx is None:
            #   select random index from range(0, num_samples - num_points)
            start_idx = np.random.randint(0, num_samples - num_points, 1)[0]
        
        # extract segment
        x = audio[start_idx: start_idx + x_len]
        y = audio[start_idx + x_len:start_idx + x_len + y_len]
        return x, y

    def _to_tensor(self, x, y=None):
        x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) < 2:
            x = torch.unsqueeze(x, 0)

        if y is not None:
            y = torch.tensor(y, dtype=torch.long)
            out = (x, y)
        else:
            out = x

        return out

    def _quantize(self, x, label=False):
        out = np.digitize(x, self.bins, right=False) - 1

        if not label:
            out = self.bins[out]

        return out
        
    def save_wav(self, filename, data, sample_rate=None, dtype=None):
        if sample_rate is None:
            sample_rate = self.sample_rate

        if dtype is None:
            dtype = self.dtype

        data = data.astype(dtype)
        return sf.write(filename, data, sample_rate)

    def label2value(self, label):
        return self.bins[label.astype(int)]

    def preprocess(self, x, y=None):
        x = self.encoder.encode(x)
        x = self._quantize(x)
        if y is not None:
            y = self.encoder.encode(y)
            y = self._quantize(y, label=True)
        if y is None:
            out = x
        else:
            out = (x, y)

        return out

    def trim_silence(self, audio, threshold=0, frame_length=2048):
        '''Removes silence at the beginning and end of a sample.'''
        if audio.size < frame_length:
            frame_length = audio.size
        frame_length = audio.shape[0] // 3
        energy = librosa.feature.rms(audio, frame_length=frame_length)
        frames = np.nonzero(energy > threshold)
        indices = librosa.core.frames_to_samples(frames)[1]

        # Note: indices can be an empty array, if the whole audio was silence.
        return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=True):
        super().__init__(RandomSampler(dataset), batch_size, drop_last)


class AudioLoader(DataLoader):
    def __init__(self, dataset, batch_size=8, drop_last=True, num_workers=1):
        sampler = AudioBatchSampler(dataset, batch_size, drop_last)
        super().__init__(dataset, batch_sampler=sampler, num_workers=num_workers)

