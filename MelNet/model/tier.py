import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gmm import sample_gmm
from .rnn import DelayedRNN
from .upsample import UpsampleRNN
import pdb

class Tier(nn.Module):
    def __init__(self, hp, freq, layers, tierN, num_class=1):
        super(Tier, self).__init__()
        num_hidden = hp.model.hidden
        self.hp = hp
        self.tierN = tierN
        self.num_class = num_class
        if self.num_class > 1:
            self.t_embedding = nn.Embedding(self.num_class, num_hidden)
            self.f_embedding = nn.Embedding(self.num_class, num_hidden)

        if(tierN == 1):
            self.W_t_0 = nn.Linear(1, num_hidden)
            self.W_f_0 = nn.Linear(1, num_hidden)
            self.W_c_0 = nn.Linear(freq, num_hidden)
            self.layers = nn.ModuleList([
                DelayedRNN(hp) for _ in range(layers)
            ])
        else:
            self.W_t = nn.Linear(1, num_hidden)
            self.layers = nn.ModuleList([
                UpsampleRNN(hp) for _ in range(layers)
            ])

        # Gaussian Mixture Model: eq. (2)
        self.K = hp.model.gmm
        self.pi_softmax = nn.Softmax(dim=3)

        # map output to produce GMM parameter eq. (10)
        self.W_theta = nn.Linear(num_hidden, 3*self.K)

    def forward(self, x, audio_lengths, class_label=None, save_hidden=False, hidden_t=None, hidden_c=None):
        # x: [B, M, T] / B=batch, M=mel, T=time
        if self.tierN == 1:
            h_t = self.W_t_0(F.pad(x, [1, -1]).unsqueeze(-1))
            h_f = self.W_f_0(F.pad(x, [0, 0, 1, -1]).unsqueeze(-1))
            h_c = self.W_c_0(F.pad(x, [1, -1]).transpose(1, 2))
            if class_label is not None:
                t_0 = self.t_embedding(class_label).view(class_label.size(0), 1, 1, -1)
                f_0 = self.f_embedding(class_label).view(class_label.size(0), 1, 1, -1)
                h_t = h_t + t_0
                h_f = h_f + f_0
            for layer in self.layers:
                h_t, h_f, h_c = self.layers[0](h_t, h_f, h_c, audio_lengths,  hidden_t, hidden_c)

            # h_t, h_f: [B, M, T, D] / D=num_hidden
            # h_c: [B, T, D]
        else:
            h_f = self.W_t(x.unsqueeze(-1))
            for layer in self.layers:
                h_f = layer(h_f, audio_lengths)

        theta_hat = self.W_theta(h_f)

        mu = theta_hat[..., :self.K] # eq. (3)
        std = theta_hat[..., self.K:2*self.K]
        pi = theta_hat[..., 2*self.K:]

        if save_hidden:
            return mu, std, pi, h_t, h_c
        else:
            return mu, std, pi
