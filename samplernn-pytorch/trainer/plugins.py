import matplotlib
matplotlib.use('Agg')

from model import Generator

import torch
from torch.autograd import Variable


from librosa.output import write_wav
from matplotlib import pyplot
from collections import defaultdict

from glob import glob
import os
import pickle
import time


class Plugin(object):
    def __init__(self, interval=None):
        if interval is None:
            interval = []
        self.trigger_interval = interval

    def register(self, trainer):
        raise NotImplementedError


class Monitor(Plugin):

    def __init__(self, running_average=True, epoch_average=True, smoothing=0.7,
                 precision=None, number_format=None, unit=''):
        '''
        :param running_average:
        :param epoch_average:
        :param smoothing:
        :param precision:
        :param number_format:
        :param unit:
        '''
        if precision is None:
            precision = 4
        if number_format is None:
            number_format = '.{}f'.format(precision)
        number_format = ':' + number_format

        super(Monitor, self).__init__([(1, 'iteration'), (1, 'epoch')])

        self.smoothing = smoothing
        self.with_running_average = running_average
        self.with_epoch_average = epoch_average

        self.log_format = number_format
        self.log_unit = unit
        self.log_epoch_fields = None
        self.log_iter_fields = ['{last' + number_format + '}' + unit]
        if self.with_running_average:
            self.log_iter_fields += [' ({running_avg' + number_format + '}' + unit + ')']
        if self.with_epoch_average:
            self.log_epoch_fields = ['{epoch_mean' + number_format + '}' + unit]

    def register(self, trainer):
        self.trainer = trainer
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['log_format'] = self.log_format
        stats['log_unit'] = self.log_unit
        stats['log_iter_fields'] = self.log_iter_fields
        if self.with_epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields
        if self.with_epoch_average:
            stats['epoch_stats'] = (0, 0)

    def iteration(self, *args):

        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['last'] = self._get_value(*args)

        if self.with_epoch_average:
            stats['epoch_stats'] = tuple(sum(t) for t in
                                         zip(stats['epoch_stats'], (stats['last'], 1)))

        if self.with_running_average:
            previous_avg = stats.get('running_avg', 0)
            stats['running_avg'] = previous_avg * self.smoothing + \
                stats['last'] * (1 - self.smoothing)

    def epoch(self, idx):

        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.with_epoch_average:

            epoch_stats = stats['epoch_stats']
            stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
            stats['epoch_stats'] = (0, 0)


class LossMonitor(Monitor):
    stat_name = 'loss'

    def _get_value(self, iteration, input, target, output, loss):
        return loss.item()


class TrainingLossMonitor(LossMonitor):

    stat_name = 'training_loss'


class Logger(Plugin):
    alignment = 4
    separator = '#' * 80

    def __init__(self, fields, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Logger, self).__init__(interval)

        self.field_widths = defaultdict(lambda: defaultdict(int))
        self.fields = list(map(lambda f: f.split('.'), fields))

    def _join_results(self, results):
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def log(self, msg):
        print(msg)

    def register(self, trainer):
        self.trainer = trainer

    def gather_stats(self):
        result = {}
        return result

    def _align_output(self, field_idx, output):
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        output = []
        name = ''
        if isinstance(stat, dict):

            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output

    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        loginfo = []

        if prefix is not None:
            loginfo.append(prefix)
            loginfo.append("\t")

        loginfo.append(output)
        if suffix is not None:
            loginfo.append("\t")
            loginfo.append(suffix)
        self.log("".join(loginfo))

    def iteration(self, *args):

        self._log_all('log_iter_fields',prefix="iteration:{}".format(args[0]))

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields',
                      prefix=self.separator + '\nEpoch summary:',
                      suffix=self.separator,
                      require_dict=True)


class ValidationPlugin(Plugin):

    def __init__(self, val_dataset, test_dataset):
        super().__init__([(1, 'epoch')])
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def register(self, trainer):
        self.trainer = trainer
        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['log_epoch_fields'] = ['{last:.4f}']
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['log_epoch_fields'] = ['{last:.4f}']

    def epoch(self, idx):
        self.trainer.model.eval()

        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['last'] = self._evaluate(self.val_dataset)
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['last'] = self._evaluate(self.test_dataset)

        self.trainer.model.train()

    def _evaluate(self, dataset):
        loss_sum = 0
        n_examples = 0
        with torch.autograd.no_grad():
            for data in dataset:
                batch_inputs = data[: -1]
                batch_target = data[-1]
                batch_size = batch_target.size()[0]

                def wrap(input):
                    if torch.is_tensor(input):
                        input = Variable(input)
                        if self.trainer.cuda:
                            input = input.cuda()
                    return input
                batch_inputs = list(map(wrap, batch_inputs))

                batch_target = Variable(batch_target)
                if self.trainer.cuda:
                    batch_target = batch_target.cuda()

                batch_output = self.trainer.model(*batch_inputs)
                loss_sum += self.trainer.criterion(batch_output, batch_target) \
                                        .data.item() * batch_size

                n_examples += batch_size

        return loss_sum / n_examples


class AbsoluteTimeMonitor(Monitor):

    stat_name = 'time'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', 's')
        kwargs.setdefault('precision', 0)
        kwargs.setdefault('running_average', False)
        kwargs.setdefault('epoch_average', False)
        super(AbsoluteTimeMonitor, self).__init__(*args, **kwargs)
        self.start_time = None

    def _get_value(self, *args):
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time


class SaverPlugin(Plugin):

    last_pattern = 'ep{}-it{}'
    best_pattern = 'best-ep{}-it{}'

    def __init__(self, checkpoints_path, keep_old_checkpoints):
        super().__init__([(1, 'epoch')])
        self.checkpoints_path = checkpoints_path
        self.keep_old_checkpoints = keep_old_checkpoints
        self._best_val_loss = float('+inf')

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        if not self.keep_old_checkpoints:
            self._clear(self.last_pattern.format('*', '*'))
        torch.save(
            self.trainer.model.state_dict(),
            os.path.join(
                self.checkpoints_path,
                self.last_pattern.format(epoch_index, self.trainer.iterations)
            )
        )

        cur_val_loss = self.trainer.stats['validation_loss']['last']
        if cur_val_loss < self._best_val_loss:
            self._clear(self.best_pattern.format('*', '*'))
            torch.save(
                self.trainer.model.state_dict(),
                os.path.join(
                    self.checkpoints_path,
                    self.best_pattern.format(
                        epoch_index, self.trainer.iterations
                    )
                )
            )
            self._best_val_loss = cur_val_loss

    def _clear(self, pattern):
        pattern = os.path.join(self.checkpoints_path, pattern)
        for file_name in glob(pattern):
            os.remove(file_name)


class GeneratorPlugin(Plugin):

    pattern = 'ep{}-s{}-c{}.wav'

    def __init__(self, samples_path, n_samples, sample_length, sample_rate, num_classes=1):
        super().__init__([(1, 'epoch')])
        self.samples_path = samples_path
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.num_classes = num_classes

    def register(self, trainer):
        self.generate = Generator(trainer.model.model, trainer.cuda)

    def epoch(self, epoch_index):
        for label in range(self.num_classes):
            samples = self.generate(self.n_samples, self.sample_length, class_label=label) \
                          .cpu().float().numpy()

            for i in range(self.n_samples):
                write_wav(
                    os.path.join(
                        self.samples_path, self.pattern.format(epoch_index, i + 1, label)
                    ),
                    samples[i, :], sr=self.sample_rate, norm=True
                )


class StatsPlugin(Plugin):

    data_file_name = 'stats.pkl'
    plot_pattern = '{}.svg'

    def __init__(self, results_path, iteration_fields, epoch_fields, plots):
        super().__init__([(1, 'iteration'), (1, 'epoch')])
        self.results_path = results_path

        self.iteration_fields = self._fields_to_pairs(iteration_fields)
        self.epoch_fields = self._fields_to_pairs(epoch_fields)
        self.plots = plots
        self.data = {
            'iterations': {
                field: []
                for field in self.iteration_fields + [('iteration', 'last')]
            },
            'epochs': {
                field: []
                for field in self.epoch_fields + [('iteration', 'last')]
            }
        }

    def register(self, trainer):
        self.trainer = trainer

    def iteration(self, *args):
        for (field, stat) in self.iteration_fields:
            self.data['iterations'][field, stat].append(
                self.trainer.stats[field][stat]
            )

        self.data['iterations']['iteration', 'last'].append(
            self.trainer.iterations
        )

    def epoch(self, epoch_index):
        for (field, stat) in self.epoch_fields:
            self.data['epochs'][field, stat].append(
                self.trainer.stats[field][stat]
            )

        self.data['epochs']['iteration', 'last'].append(
            self.trainer.iterations
        )

        data_file_path = os.path.join(self.results_path, self.data_file_name)
        with open(data_file_path, 'wb') as f:
            pickle.dump(self.data, f)

        for (name, info) in self.plots.items():
            x_field = self._field_to_pair(info['x'])

            try:
                y_fields = info['ys']
            except KeyError:
                y_fields = [info['y']]

            labels = list(map(
                lambda x: ' '.join(x) if type(x) is tuple else x,
                y_fields
            ))
            y_fields = self._fields_to_pairs(y_fields)

            try:
                formats = info['formats']
            except KeyError:
                formats = [''] * len(y_fields)

            pyplot.gcf().clear()

            for (y_field, format, label) in zip(y_fields, formats, labels):
                if y_field in self.iteration_fields:
                    part_name = 'iterations'
                else:
                    part_name = 'epochs'

                xs = self.data[part_name][x_field]
                ys = self.data[part_name][y_field]

                pyplot.plot(xs, ys, format, label=label)

            if 'log_y' in info and info['log_y']:
                pyplot.yscale('log')

            pyplot.legend()
            pyplot.savefig(
                os.path.join(self.results_path, self.plot_pattern.format(name))
            )

    @staticmethod
    def _field_to_pair(field):
        if type(field) is tuple:
            return field
        else:
            return (field, 'last')

    @classmethod
    def _fields_to_pairs(cls, fields):
        return list(map(cls._field_to_pair, fields))


class CometPlugin(Plugin):

    def __init__(self, experiment, fields):
        super().__init__([(1, 'epoch')])

        self.experiment = experiment
        self.fields = [
            field if type(field) is tuple else (field, 'last')
            for field in fields
        ]

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        for (field, stat) in self.fields:
            self.experiment.log_metric(field, self.trainer.stats[field][stat])
        self.experiment.log_epoch_end(epoch_index)
