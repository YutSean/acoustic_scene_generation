import utils

import torch
import numpy as np
from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
)

from librosa.core import load
from librosa.util import fix_length
from natsort import natsorted

from os import listdir
from os.path import join
import os
import pdb


class FolderDataset(Dataset):

    def __init__(self, path, overlap_len, q_levels, ratio_min=0, ratio_max=1, sample_rate=16000, audio_length=160000):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.class_mapping = {}
        self.path = path
        dirs = listdir(path)
        for dir in dirs:
            if not os.path.isdir(join(join(path, dir))):
                dirs.remove(dir)
        if len(dirs) < 1:
            raise NotADirectoryError("The dataset path need subdirectories.")
        for idx, dir in enumerate(dirs):
            self.class_mapping[dir] = idx

        file_names = []
        for dir in dirs:
            files = listdir(join(path, dir))
            for file in files:
                file_names.append((join(join(path, dir), file), self.class_mapping[dir]))
        file_names = natsorted(file_names)
        # file_names = natsorted(
        #     [join(path, file_name) for file_name in listdir(path)]
        # )
        self.file_names = file_names[
            int(ratio_min * len(file_names)): int(ratio_max * len(file_names))
        ]

    def __getitem__(self, index):
        file_path, class_label = self.file_names[index]
        (seq, _) = load(file_path, sr=self.sample_rate, mono=True)
        seq = fix_length(seq, size=self.audio_length, mode='edge')
        return torch.cat((
            torch.LongTensor([class_label]),
            # torch.LongTensor(self.overlap_len)
            #      .fill_(utils.q_zero(self.q_levels)),
            utils.linear_quantize(
                torch.from_numpy(seq), self.q_levels
            )
        ))

    def getClassSplit(self, class_num=0, seq_len=64):
        file_path = ''
        for key in self.class_mapping:
            if self.class_mapping[key] == class_num:
                file_path = join(self.path, key)
        files = listdir(file_path)
        result = None

        pick_one = files[np.random.randint(0, len(files))]
        pick_one = join(file_path, pick_one)
        seq, _ = load(pick_one, sr=self.sample_rate, mono=True)
        seq = fix_length(seq, size=self.audio_length, mode='edge')

        while result is None or len(result) != seq_len:
            start_idx = np.random.randint(0, len(seq - seq_len - 1))
            result = seq[start_idx: start_idx + seq_len]

        return result

    def __len__(self):
        return len(self.file_names)


class DataLoader(DataLoaderBase):

    def __init__(self, dataset, batch_size, seq_len, overlap_len,
                 *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len

    def __iter__(self):
        for batch in super().__iter__():
            (batch_size, n_samples) = batch.size()
            class_labels = batch[:, 0]
            batch = batch[:, 1:]
            reset = True
            batch_len = batch.size(-1)

            for seq_begin in range(self.overlap_len, n_samples, self.seq_len):
                from_index = seq_begin - self.overlap_len   # 0
                to_index = seq_begin + self.seq_len         # 16 + 1024 = 1040
                if to_index >= batch_len:
                    break
                sequences = batch[:, from_index: to_index]  # [0, 1040]
                input_sequences = sequences[:, :-1]         # [0, 1039]
                target_sequences = sequences[:, -self.seq_len:].contiguous()

                # mlp_input = torch.zeros(batch_size, self.seq_len, self.overlap_len)
                # for b in range(batch_size):
                #     for idx in range(self.seq_len):
                #         mlp_input[:, idx, :] = input_sequences[:, idx: idx + self.overlap_len]

                yield (input_sequences, reset, class_labels, target_sequences)

                reset = False

    def __len__(self):
        raise NotImplementedError()
        # return len(self.dataset)
