import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from .tier import Tier
from .tts import TTS
from .loss import GMMLoss
from text import text_to_sequence
from utils.gmm import sample_gmm
from utils.constant import f_div, t_div
from utils.hparams import load_hparam_str
from utils.tierutil import TierUtil
import pdb

class MelNet(nn.Module):
    def __init__(self, hp, args, infer_hp):
        super(MelNet, self).__init__()
        self.hp = hp
        self.args = args
        self.infer_hp = infer_hp
        self.f_div = f_div[hp.model.tier + 1]
        self.t_div = t_div[hp.model.tier]
        self.n_mels = hp.audio.n_mels

        self.tierutil = TierUtil(hp)

        if infer_hp.conditional:
            self.tiers = [TTS(
                    hp=hp,
                    freq=hp.audio.n_mels // self.f_div * f_div[1],
                    layers=hp.model.layers[0]
                )] + [Tier(
                    hp=hp,
                    freq=hp.audio.n_mels // self.f_div * f_div[tier],
                    layers=hp.model.layers[tier - 1],
                    tierN=tier
                ) for tier in range(2, hp.model.tier + 1)]
        else:
            self.tiers = [Tier(
                    hp=hp,
                    freq=hp.audio.n_mels // self.f_div * f_div[tier],
                    layers=hp.model.layers[tier-1],
                    tierN=tier,
                    num_class=10
                ) for tier in range(1, hp.model.tier + 1)]
        self.tiers = nn.ModuleList(
            [None] + [nn.DataParallel(tier).cuda() for tier in self.tiers]
        )

    def forward(self, x, tier_num):
        assert tier_num > 0, 'tier_num should be larger than 0, got %d' % tier_num

        return self.tiers[tier_num](x)

    def sample(self, condition):
        x = None
        if condition is not None:
            # seq = torch.from_numpy(text_to_sequence(condition)).long().unsqueeze(0)
            x = condition
        else:
            seq = torch.LongTensor([[0]])
        # input_lengths = torch.LongTensor([seq[0].shape[0]]).cuda()
        if x is not None:
            audio_lengths = torch.LongTensor([x.size()[-1]]).cuda()
        else:
            audio_lengths = torch.LongTensor([0]).cuda()
        ## Tier 1 ##
        tqdm.write('Tier 1')
        if self.args.timestep == 0:
            mu, std, pi = self.tiers[1](x, audio_lengths)
            temp = sample_gmm(mu, std, pi)
            return temp

        for t in tqdm(range(self.args.timestep // self.t_div)):
            audio_lengths += 1
            if x is None:
                x = torch.zeros((1, self.n_mels // self.f_div, 1)).cuda()
            else:
                x = torch.cat([x, torch.zeros((1, self.n_mels // self.f_div, 1)).cuda()], dim=-1)
            for m in tqdm(range(self.n_mels // self.f_div)):
                torch.cuda.synchronize()
                if self.infer_hp.conditional:
                    # mu, std, pi, _ = self.tiers[1](x, seq, input_lengths, audio_lengths)
                    break
                else:
                    mu, std, pi = self.tiers[1](x, audio_lengths)
                temp = sample_gmm(mu, std, pi)
                new_idx = audio_lengths.item() - 1
                x[:, m, new_idx] = temp[:, m, new_idx]

        ## Tier 2~N ##
        for tier in tqdm(range(2, self.hp.model.tier + 1)):
            tqdm.write('Tier %d' % tier)
            mu, std, pi = self.tiers[tier](x)
            temp = sample_gmm(mu, std, pi)
            x = self.tierutil.interleave(x, temp, tier + 1)

        return x

    def load_tiers(self):
        for idx, chkpt_path in enumerate(self.infer_hp.checkpoints):
            checkpoint = torch.load(chkpt_path)

            hp = load_hparam_str(checkpoint['hp_str'])

            if self.hp != hp:
                print('Warning: hp different in file %s' % chkpt_path)
            
            self.tiers[idx+1].load_state_dict(checkpoint['model'])

    def sample_dependence(self, condition, label, dependence_length):
        x = None
        if condition is not None:
            # seq = torch.from_numpy(text_to_sequence(condition)).long().unsqueeze(0)
            x = condition
        else:
            seq = torch.LongTensor([[0]])
        if x is not None:
            audio_lengths = torch.LongTensor([x.size()[-1]]).cuda()
        else:
            audio_lengths = torch.LongTensor([0]).cuda()
        for t in tqdm(range(self.args.timestep // self.t_div)):
            # audio_lengths += 1
            if x is None:
                x = torch.zeros((1, self.n_mels // self.f_div, 1)).cuda()
            else:
                x = torch.cat([x, torch.zeros((1, self.n_mels // self.f_div, 1)).cuda()], dim=-1)
            for m in tqdm(range(self.n_mels // self.f_div)):
                torch.cuda.synchronize()
                if self.infer_hp.conditional:
                    # mu, std, pi, _ = self.tiers[1](x, seq, input_lengths, audio_lengths)
                    break
                else:
                    class_label = torch.tensor(label, dtype=torch.long) if isinstance(label, int) else torch.LongTensor(label)
                    if m == 0:
                        mu, std, pi, h_t, h_c = self.tiers[1](x[:, :, -dependence_length:], audio_lengths, class_label.cuda(non_blocking=True).unsqueeze(0), save_hidden=True, hidden_t=None, hidden_c=None)
                    else:
                        mu, std, pi = self.tiers[1](x[:, :, -dependence_length:], audio_lengths, class_label.cuda(non_blocking=True).unsqueeze(0), save_hidden=False, hidden_t=h_t, hidden_c=h_c)
                temp = sample_gmm(mu, std, pi)
                new_idx = audio_lengths.item() - 1
                x[:, m, -1] = temp[:, m, new_idx]

        return x
