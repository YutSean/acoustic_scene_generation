import nn
import utils

import torch
from torch.nn import functional as F
from torch.nn import init

import numpy as np
import pdb


class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels,
                 weight_norm, num_classes=10):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels
        self.num_classes = num_classes

        # ns_frame_samples = np.cumprod(frame_sizes).tolist()
        ns_frame_samples = [64]

        # ns_frame_samples = ns_frame_samples[1:]
        # frame_sizes = frame_sizes[1:]
        self.frame_level_rnns = torch.nn.ModuleList([
            FrameLevelRNN(
                frame_size, n_frame_samples, n_rnn, dim, learn_h0, weight_norm, num_classes=self.num_classes
            )
            for (frame_size, n_frame_samples) in zip(
                frame_sizes, ns_frame_samples
            )
        ])

        self.sample_level_mlp = SampleLevelMLP(
            frame_sizes[0], dim, q_levels, weight_norm
        )

    @property
    def lookback(self):
        # return self.frame_level_rnns[-1].n_frame_samples
        return self.frame_level_rnns[-1].frame_size


class FrameLevelRNN(torch.nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim,
                 learn_h0, weight_norm, skip_connection=False, num_classes=0, embedding_dim=256):
        super().__init__()

        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim
        self.skip_connection = skip_connection
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        h0 = torch.zeros(n_rnn, dim)
        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)
        else:
            self.register_buffer('h0', torch.autograd.Variable(h0))

        # self.input_expand = torch.nn.Conv1d(
        #     in_channels=n_frame_samples,
        #     out_channels=dim,
        #     kernel_size=1
        # )
        # init.kaiming_uniform_(self.input_expand.weight)
        # init.constant_(self.input_expand.bias, 0)
        # if weight_norm:
        #     self.input_expand = torch.nn.utils.weight_norm(self.input_expand)

        # self.rnn = torch.nn.GRU(
        #     input_size=dim,
        #     hidden_size=dim,
        #     num_layers=n_rnn,
        #     batch_first=True
        # )

        self.rnn = torch.nn.GRU(
            input_size=self.frame_size + self.num_classes,
            hidden_size=dim,
            num_layers=n_rnn,
            batch_first=True
        )
        for i in range(n_rnn):
            nn.concat_init(
                getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, nn.lecun_uniform]
            )
            init.constant_(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)

            nn.concat_init(
                getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, init.orthogonal_]
            )
            init.constant_(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)

        # self.upsampling = nn.LearnedUpsampling1d(
        #     in_channels=dim,
        #     out_channels=dim,
        #     kernel_size=frame_size
        # )
        self.rnns_out = torch.nn.Linear(self.dim, self.frame_size * self.dim)
        # init.uniform_(
        #     self.upsampling.conv_t.weight, -np.sqrt(6 / dim), np.sqrt(6 / dim)
        # )
        # init.constant_(self.upsampling.bias, 0)
        # if weight_norm:
        #     self.upsampling.conv_t = torch.nn.utils.weight_norm(
        #         self.upsampling.conv_t
        #     )
        if weight_norm:
            self.rnns_out = torch.nn.utils.weight_norm(self.rnns_out)

    def forward(self, prev_samples, upper_tier_conditioning, hidden, class_label=0):
        (batch_size, _, _) = prev_samples.size()

        # input = self.input_expand(
        #   prev_samples.permute(0, 2, 1)
        # ).permute(0, 2, 1)
        input = prev_samples

        if upper_tier_conditioning is not None:
            input += upper_tier_conditioning

        reset = hidden is None

        if hidden is None:
            (n_rnn, _) = self.h0.size()
            hidden = self.h0.unsqueeze(1) \
                            .expand(n_rnn, batch_size, self.dim) \
                            .contiguous()
        if self.num_classes > 0:
            batch_num = input.size(0)
            if not isinstance(class_label, torch.LongTensor) or not isinstance(class_label, torch.cuda.LongTensor):
                # class_label = torch.tensor(class_label, dtype=torch.long).detach()
                class_label = class_label.long()
                if torch.cuda.is_available():
                    class_label = class_label.cuda()
            addition = F.one_hot(class_label % self.num_classes, num_classes=self.num_classes)
            addition = addition.repeat(batch_size, input.size(1), 1).float()
            input = torch.cat([input, addition], dim=-1)
        (output, hidden) = self.rnn(input, hidden)
        # if self.skip_connection:
        #     output += input

        # output = self.upsampling(
        #     output.permute(0, 2, 1)
        # ).permute(0, 2, 1)

        return self.rnns_out(output), hidden


class SampleLevelMLP(torch.nn.Module):

    def __init__(self, frame_size, dim, q_levels, weight_norm):
        super().__init__()

        self.q_levels = q_levels
        self.frame_size = frame_size
        self.dim = dim

        self.embedding = torch.nn.Embedding(
            self.q_levels,
            self.q_levels
        )
        self.last_out_shape = q_levels

        # self.input = torch.nn.Conv1d(
        #     in_channels=q_levels,
        #     out_channels=dim,
        #     kernel_size=frame_size,
        #     bias=False
        # )
        self.input = torch.nn.Linear(self.frame_size * self.last_out_shape, self.dim, bias=False)
        init.kaiming_uniform_(self.input.weight)
        if weight_norm:
            self.input = torch.nn.utils.weight_norm(self.input)

        # self.hidden = torch.nn.Conv1d(
        #     in_channels=dim,
        #     out_channels=dim,
        #     kernel_size=1
        # )
        self.hidden = torch.nn.Linear(self.dim, self.dim)

        init.kaiming_uniform_(self.hidden.weight)
        init.constant_(self.hidden.bias, 0)
        if weight_norm:
            self.hidden = torch.nn.utils.weight_norm(self.hidden)

        self.hidden_2 = torch.nn.Linear(self.dim, self.dim)
        init.kaiming_uniform_(self.hidden_2.weight)
        init.constant_(self.hidden_2.bias, 0)

        # self.output = torch.nn.Conv1d(
        #     in_channels=dim,
        #     out_channels=q_levels,
        #     kernel_size=1
        # )
        self.output = torch.nn.Linear(self.dim, self.q_levels)
        nn.lecun_uniform(self.output.weight)
        init.constant_(self.output.bias, 0)
        if weight_norm:
            self.output = torch.nn.utils.weight_norm(self.output)

    def forward(self, prev_samples, upper_tier_conditioning):
        (batch_size, _, _) = upper_tier_conditioning.size()
        prev_samples = prev_samples.unfold(1, self.frame_size, 1)
        prev_samples = self.embedding(
            prev_samples.contiguous().view(-1)
        ).view(
            batch_size, -1, self.q_levels
        )
        # prev_samples = prev_samples.permute(0, 2, 1)
        # upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)
        prev_samples = prev_samples.view(batch_size, -1, self.frame_size * self.last_out_shape)
        upper_tier_conditioning = upper_tier_conditioning.view(batch_size, -1, self.dim)
        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden_2(x))
        # x = self.output(x).permute(0, 2, 1).contiguous()
        x = self.output(x)

        return F.log_softmax(x.view(-1, self.q_levels)) \
                .view(batch_size, -1, self.q_levels)


class Runner:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.model.frame_level_rnns}

    def run_rnn(self, rnn, prev_samples, upper_tier_conditioning, class_label):
        (output, new_hidden) = rnn(
            prev_samples, upper_tier_conditioning, self.hidden_states[rnn], class_label
        )
        self.hidden_states[rnn] = new_hidden.detach()
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model):
        super().__init__(model)
        self.num_classes = model.num_classes

    def forward(self, input_sequences, reset, z=None):
        if reset:
            self.reset_hidden_states()

        (batch_size, _) = input_sequences.size()

        if z is None:
            upper_tier_conditioning = None
        else:
            upper_tier_conditioning = None

        # else:
            # upper_tier_conditioning = self.model.class_embedding(z).view(batch_size, 1, -1) + self.model.class_bias
        # for rnn in reversed(self.model.frame_level_rnns):
        #     # from_index = self.model.lookback - rnn.n_frame_samples
        #     from_index = 0
        #     to_index = -rnn.n_frame_samples + 1
        #     prev_samples = 2 * utils.linear_dequantize(
        #         input_sequences[:, from_index: to_index],
        #         self.model.q_levels
        #     )
        #     # if (len(prev_samples.flatten())) < batch_size * rnn.n_frame_samples:
        #     prev_samples = prev_samples.contiguous().view(
        #         batch_size, -1, rnn.frame_size
        #     )
        #     upper_tier_conditioning = self.run_rnn(
        #         rnn, prev_samples, upper_tier_conditioning, z
        #     )
        rnn = self.model.frame_level_rnns[0]
        prev_samples = 2 * utils.linear_dequantize(input_sequences[:, :-rnn.frame_size + 1], self.model.q_levels)
        prev_samples = prev_samples.contiguous().view(
            batch_size, -1, rnn.frame_size
        )
        upper_tier_conditioning = self.run_rnn(
            rnn, prev_samples, upper_tier_conditioning, z
        )
        # bottom_frame_size = self.model.frame_level_rnns[0].frame_size
        # mlp_input_sequences = input_sequences[:, self.model.lookback - bottom_frame_size:]
        # mlp_input_sequences = input_sequences[:, :]
        # seq_len = mlp_input_sequences.size(-1) - bottom_frame_size
        # mlp_input = torch.zeros(batch_size, seq_len, bottom_frame_size)
        # for idx in range(seq_len):
        #     mlp_input[:, idx, :] = mlp_input_sequences[:, idx: idx + bottom_frame_size]

        return self.model.sample_level_mlp(
            input_sequences, upper_tier_conditioning
        )


class Generator(Runner):

    def __init__(self, model, cuda=False):
        super().__init__(model)
        self.cuda = cuda
        self.model = model.cuda()

    def __call__(self, n_seqs, seq_len, class_label=0, data_seed=None):
        # generation doesn't work with CUDNN for some reason

        torch.backends.cudnn.enabled = False
        label_tensor = torch.LongTensor([class_label])
        with torch.no_grad():
            self.reset_hidden_states()

            # bottom_frame_size = self.model.frame_level_rnns[0].n_frame_samples
            bottom_frame_size = 16
            sequences = torch.LongTensor(n_seqs, self.model.lookback + seq_len) \
                             .fill_(utils.q_zero(self.model.q_levels))
            if data_seed is not None:
                seeds = []
                for _ in range(n_seqs):
                    seeds.append(utils.linear_quantize(torch.from_numpy(data_seed.getClassSplit(class_num=class_label, seq_len=self.model.lookback)), self.model.q_levels))
                seed = torch.stack(seeds)
                sequences[:, :self.model.lookback] = seed
            frame_level_outputs = [None for _ in self.model.frame_level_rnns]

            for i in range(self.model.lookback, self.model.lookback + seq_len):
                for (tier_index, rnn) in \
                        reversed(list(enumerate(self.model.frame_level_rnns))):
                    # if i % rnn.n_frame_samples != 0:
                    #     continue

                    prev_samples = sequences[:, i - 16:i]
                    prev_samples = torch.autograd.Variable(
                        2 * utils.linear_dequantize(
                            prev_samples,
                            self.model.q_levels
                        ).unsqueeze(1)
                    )
                    if self.cuda:
                        prev_samples = prev_samples.cuda()
                        label_tensor = label_tensor.cuda()

                    if tier_index == len(self.model.frame_level_rnns) - 1:
                        upper_tier_conditioning = None
                        # if self.model.num_classes > 1:
                        #     upper_tier_conditioning = self.model.class_embedding(label_tensor) + self.model.class_bias.cuda()
                        #     if self.cuda:
                        #         upper_tier_conditioning = upper_tier_conditioning.cuda()
                    else:
                        frame_index = (i // rnn.n_frame_samples) % \
                            self.model.frame_level_rnns[tier_index + 1].frame_size
                        upper_tier_conditioning = \
                            frame_level_outputs[tier_index + 1][:, frame_index, :] \
                                               .unsqueeze(1)
                    if isinstance(class_label, int):
                        class_label = torch.Tensor([class_label])
                    frame_level_outputs[tier_index] = self.run_rnn(
                        rnn, prev_samples, upper_tier_conditioning, class_label
                    )
                prev_samples = torch.autograd.Variable(
                    sequences[:, i - bottom_frame_size: i]
                )
                if self.cuda:
                    prev_samples = prev_samples.cuda()
                frame_level_outputs[0] = frame_level_outputs[0].view(n_seqs, bottom_frame_size, -1)
                upper_tier_conditioning = \
                    frame_level_outputs[0][:, i % bottom_frame_size, :].unsqueeze(1)
                sample_dist = self.model.sample_level_mlp(
                    prev_samples, upper_tier_conditioning
                ).squeeze(1).exp_().data
                sequences[:, i] = sample_dist.multinomial(1).squeeze(1)

            torch.backends.cudnn.enabled = True

            return sequences[:, self.model.lookback:]
