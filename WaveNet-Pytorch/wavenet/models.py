import time, copy
from collections import OrderedDict
from functools import reduce
from wavenet.earlystopping import EarlyStopping
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import visdom

import pdb


class Model(nn.Module):
    """
        A Complete Wavenet Model
        Args:
            layers (Int):               Number of layers in each block
            blocks (Int):               Number of wavenet blocks of this model
            dilation_channels (Int):    Number of channels for the dilated convolution
            residual_channels (Int):    Number of channels for the residual connection
            skip_channels (Int):        Number of channels for the skip connections
            classes (Int):              Number of possible values each sample can have
            output_length (Int):        Number of samples that are generated for each input
            kernel_size (Int):          Size of the dilation kernel
            dtype:                      Parameter type of this model
        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`
            - Output: :math:`()`
            L should be the length of the receptive field
        """
    def __init__(self, 
                 layers=10,
                 blocks=5,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=32,
                 end_channels=256,
                 num_classes=256,
                 output_length=1,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 num_scenes=10,
                 bias=True,
                 gc=False):
        super(Model, self).__init__()
        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.output_length = output_length
        self.end_channels = end_channels
        self.gc = gc
        self.num_scenes = num_scenes

        # if self.gc:
        self.gc_size = 256
        self.gc_embedding = nn.Embedding(self.num_scenes, self.gc_size)
        self.set_device()

        # build model
        self.receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []

        self.net = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.num_classes,
                                    out_channels=residual_channels,
                                    kernel_size=2,
                                    bias=False)

        self.net = nn.ModuleList()
        # for _ in range(blocks):
        #     additional_scope = kernel_size - 1
        #     # new_dilation = 1
        #     for i in range(layers):
        #         self.receptive_field += additional_scope
        #         additional_scope *= 2
                # new_dilation *= 2
        self.receptive_field = blocks * (2 ** layers) + 1
        for b in range(blocks):
            # additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.net.append(GatedResidualBlock(self.residual_channels, self.dilation_channels, 2,
                                                   self.output_length, dilation=new_dilation, gc=self.gc,
                                                   gc_size=self.gc_size, receptive_field=self.receptive_field,
                                                   skip_channel=self.skip_channels,
                                                   residual_channel=self.residual_channels,
                                                   num_scenes=self.num_scenes))

                # self.receptive_field += additional_scope
                # additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=self.skip_channels, out_channels=self.end_channels, kernel_size=1)
        self.relu_1 = nn.LeakyReLU()
        self.end_conv_2 = nn.Conv1d(in_channels=self.end_channels, out_channels=self.num_classes, kernel_size=1)
        self.relu_2 = nn.LeakyReLU()

    def forward(self, x, label):
        skips = []
        condition = None
        x = self.start_conv(x)
        # for layer, batch_norm in zip(self.net, self.batch_norms):
        #     x, skip = layer(x)
        #     x = batch_norm(x)
        #     skips.append(skip)
        if label is not None:
            condition = self.gc_embedding(label.view(-1, 1))
        for layer in self.net:
            x, skip = layer(x, condition)
            skips.append(skip)

        x = reduce((lambda a, b: torch.add(a, b)), skips)
        x = self.relu_1(self.end_conv_1(x))
        x = self.relu_2(self.end_conv_2(x))
        # return self.h_class(x)
        return x

    def init_weights(self, m):
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def set_device(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


    def valid(self, dataloader):
        losses = []

        for inputs, labels in dataloader:
            inputs = self._one_hot(inputs)
            inputs.transpose_(-2, -1)

            inputs = inputs.float()
            labels = labels.view(-1)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            output = self(inputs).view(-1, self.num_classes)
            batch_loss = self.criterion(output, labels)
            losses.append(batch_loss.item())
        return sum(losses) / len(losses)

    def _one_hot(self, data):
        if not isinstance(data, torch.LongTensor) and not isinstance(data, torch.cuda.LongTensor):
            raise ValueError('Type of input tensor must be LongTensor.')
        encoded = F.one_hot(data, num_classes=self.num_classes)

        # output = encoded.view(data.size(0), -1, self.num_classes)
        output = encoded.squeeze()
        return output

    def train(self,
              dataloader,
              num_epochs=25,
              validation=False,
              disp_interval=None,
              use_visdom=False,
              model_dir=None,
              start_epoch=0,
              num_steps=100):
        if model_dir is None:
            model_dir = './'
        since = time.time()
        self.to(self.device)

        if validation:
            phase = 'Validation'
        else:
            phase = 'Training'

        if use_visdom:
            vis = visdom.Visdom()
            gen = Generator(self, dataloader.dataset)
        else:
            vis = None
        es = EarlyStopping(patience=10)
        steps = 0
        losses = []
        # for epoch in range(start_epoch + 1, num_epochs + start_epoch + 1):
        while True:
            if not validation:
                # self.scheduler.step()
                super().train()
            else:
                self.eval()

            # reset loss for current phase and epoch
            running_loss = 0.0
            running_corrects = 0
            for segments, targets, labels in dataloader:
                if steps > num_steps:
                    break
                inputs = self._one_hot(segments)
                inputs = inputs.transpose(1, 2)

                inputs = inputs.float()
                labels = labels.view(-1, 1)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                # track history only during training phase
                with torch.set_grad_enabled(not validation):
                    outputs = self(inputs, labels).transpose(1, 2)
                    outputs.squeeze_()
                    targets.squeeze_()
                    loss = self.criterion(outputs, targets)
                    # print("Batch loss is {}".format(loss.item()))

                    if not validation:
                        loss.backward()
                        self.optimizer.step()
                    steps += 1
                    if disp_interval is not None and steps % disp_interval == 0:

                        print("Finished {}/{} steps with loss: {}".format(steps, num_steps, loss.item()))
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir, exist_ok=True)
                        model_name = 'model_{}.pt'.format(steps)
                        torch.save(self, os.path.join(model_dir, model_name))

                        # self.scheduler.step()
                    # print("Batch loss: {}".format(loss.item()))
                running_loss += loss.item() * inputs.size(0)

            losses.append(running_loss)
            # if validation_data is not None:
            #     metric = self.valid(validation_data)
            #     if es.step(metric):
            #         break
            # metric = self.valid(dataloader)
            # if es.step(metric):
            #     break
            # if disp_interval is not None and steps % disp_interval == 0:
            #     epoch_loss = running_loss / len(dataloader.dataset)
            #     print('Steps {} / {}'.format(steps, num_steps))
            #     print('Learning Rate: {}'.format(self.scheduler.get_lr()))
            #     print('{} Loss: {}'.format(phase, epoch_loss))
            #     print('-' * 10)
            #     print()
            #     if not os.path.exists(model_dir):
            #         os.makedirs(model_dir, exist_ok=True)
            #     model_name = 'model_{}.pt'.format(steps)
            #     torch.save(self, model_dir + model_name)
            #     print('Saving model data to file: {}'.format(model_name))
            #
            #     if vis is not None:
            #         # display network weights
            #         for m in self.hs:
            #             _vis_hist(vis, m.gatedconv.conv_f.weight,
            #                       m.name + ' ' + 'W-conv_f')
            #             _vis_hist(vis, m.gatedconv.conv_g.weight,
            #                       m.name + ' ' + 'W-conv_g')
            #             _vis_hist(vis, m.gatedconv.conv_f.bias,
            #                       m.name + ' ' + 'b-conv_f')
            #             _vis_hist(vis, m.gatedconv.conv_g.bias,
            #                       m.name + ' ' + 'b-conv_g')
            #
            #         # display raw outputs
            #         _vis_hist(vis, outputs, 'Outputs')
            #
            #         # display loss over time
            #         _vis_plot(vis, np.array(losses) / len(dataloader), 'Losses')
            #
            #         # display audio sample
            #         _vis_audio(vis, gen, inputs[0], 'Sample Audio', n_samples=44100,
            #                    sample_rate=dataloader.dataset.sample_rate)
            if steps > num_steps:
                model_name = 'model_{}.pt'.format(steps)
                torch.save(self, model_dir + model_name)
                print("Finish {} steps and saved the model.".format(steps))
                break


def _flatten(t):
    t = t.to(torch.device('cpu'))
    return t.data.numpy().reshape([-1])


def _vis_hist(vis, t, title):
    vis.histogram(_flatten(t), win=title, opts={'title': title})


def _vis_plot(vis, t, title):
    vis.line(t, X=np.array(range(len(t))), win=title, opts={'title': title})


def _vis_audio(vis, gen, t, title, n_samples=50, sample_rate=44100):
    y = gen.run(t, n_samples, disp_interval=10).reshape([-1])
    t = gen.dataset.encoder.decode(gen.tensor2numpy(t.cpu())).reshape([-1])
    audio = np.concatenate((t, y))
    vis.audio(audio, win=title, 
              opts={'title': title, 'sample_frequency': sample_rate})


class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, num_scenes=0, gc=False, gc_size=256, receptive_field=512):
        super(GatedConv1d, self).__init__()
        self.num_scenes = num_scenes
        self.gc = gc
        self.gc_size = gc_size
        self.dilation = dilation
        self.receptive_field = receptive_field
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        if self.gc:
            # self.embedding = nn.Embedding(self.num_scenes, self.gc_size)
            self.f_projection = nn.Linear(self.gc_size, self.receptive_field, bias=False)
            self.g_projection = nn.Linear(self.gc_size, self.receptive_field, bias=False)

    def forward(self, x, label=None):
        # padding = self.dilation - (x.shape[-1] + self.dilation - 1) % self.dilation
        x = nn.functional.pad(x, [self.dilation, 0])
        if label is not None:
            # label = label.view(-1, 1)
            f_p = self.f_projection(label)
            g_p = self.g_projection(label)

            out = torch.mul(self.tanh(self.conv_f(x) + f_p), self.sig(self.conv_g(x) + g_p))
        else:
            out = torch.mul(self.tanh(self.conv_f(x)), self.sig(self.conv_g(x)))
        return out


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_width, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, skip_channel=512, residual_channel=32, gc=False, gc_size=256,
                 receptive_field=512, num_scenes=0):
        super(GatedResidualBlock, self).__init__()

        self.skip_channel = skip_channel
        self.residual_channel = residual_channel
        self.output_width = output_width

        self.gc = gc
        self.gc_size = gc_size
        self.receptive_field = receptive_field
        self.num_scenes = num_scenes

        self.gatedconv = GatedConv1d(in_channels, out_channels, kernel_size, 
                                     stride=stride, padding=padding, 
                                     dilation=dilation, groups=groups, bias=bias, gc=gc, gc_size=gc_size,
                                     receptive_field=receptive_field - 1, num_scenes=self.num_scenes)
        self.conv_residual = nn.Conv1d(out_channels, residual_channel, 1, stride=1, padding=0,
                                       dilation=1, groups=1, bias=bias)
        self.conv_skip = nn.Conv1d(out_channels, self.skip_channel, kernel_size=1, stride=1, padding=0,
                                   dilation=1, bias=bias)

    def forward(self, x, label=None):
        if self.gc:
            output = self.gatedconv(x, label)
        else:
            output = self.gatedconv(x)
        skip = self.conv_skip(output)
        residual = torch.add(self.conv_residual(output), x)
        # residual = torch.add(skip, x)
        skip_cut = skip.shape[-1] - self.output_width
        skip = skip.narrow(-1, skip_cut, self.output_width)
        return residual, skip


class Generator(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def _shift_insert(self, x, y):
        x = x.narrow(-1, y.shape[-1], x.shape[-1] - y.shape[-1])
        dims = [1] * len(x.shape)
        dims[-1] = y.shape[-1]
        y = y.reshape(dims)
        return torch.cat([x, self.dataset._to_tensor(y)], -1)

    def tensor2numpy(self, x):
        return x.data.numpy()

    def predict(self, x, label):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1)
        x = self.model._one_hot(x.long())
        x = x.transpose(1, 2)
        x = x.to(self.model.device)
        output = self.model(x.float(), label).transpose(1, 2)
        output = output.argmax(2).view(-1)
        return output

    def run(self, seed, num_samples, gc=None, y_len=1, disp_interval=None, label=0):
        insert_point = self.model.receptive_field
        batch_size = seed.size(0)
        seed = seed.view(batch_size, -1)
        label = torch.Tensor([label] * batch_size).view(1, -1).long().to(self.model.device)
        with torch.no_grad():
            if gc == None:
                x = torch.zeros(batch_size, num_samples).long().to(self.model.device)
                x = torch.cat((seed, x), dim=-1)
            else:
                if len(gc) != self.model.receptive_field:
                    raise ValueError("The length of global condition does't match.")
                x = torch.cat((gc, torch.zeros(num_samples)), 0)
            while insert_point < self.model.receptive_field + num_samples:
                x[:, insert_point: insert_point + y_len] = self.predict(x[:, insert_point - self.model.receptive_field: insert_point], label).view(batch_size, -1)
                if insert_point - self.model.receptive_field % disp_interval == 0:
                    print('Finish {}/{}'.format(insert_point - self.model.receptive_field + 1, num_samples))
                print('Finish {} steps.'.format(insert_point - self.model.receptive_field))
                insert_point += y_len
        # move output from [0, 256] to [-128, 127]
        out = x[:, self.model.receptive_field:] - 128
        out = librosa.mu_expand(out.cpu().numpy(), quantize=True)

        return out



    # def run(self, num_samples, gc=None, y_len=1, disp_interval=None):
    #
    #     # x = self.dataset._to_tensor(self.dataset.preprocess(x))
    #     # x = torch.unsqueeze(x, 0)
    #     receiptive_field = (0, self.model.receptive_field)
    #     x = torch.zeros(num_samples + receiptive_field, 1)
    #
    #     # y_len = self.dataset.y_len
    #     out = np.zeros((num_samples // y_len + 1) * y_len)
    #     n_predicted = 0
    #     for i in range(num_samples // y_len + 1):
    #         if disp_interval is not None and i % disp_interval == 0:
    #             print('Sample {} / {}'.format(i * y_len, num_samples))
    #
    #         y_i = self.tensor2numpy(self.predict(x).cpu())
    #         y_i = self.dataset.label2value(y_i.argmax(axis=1))[0]
    #         y_decoded = self.dataset.encoder.decode(y_i)
    #
    #         out[n_predicted:n_predicted + len(y_decoded)] = y_decoded
    #         n_predicted += len(y_decoded)
    #
    #         # shift sequence and insert generated va
    #         x = self._shift_insert(x, y_i)
    #     return out[:num_samples]
