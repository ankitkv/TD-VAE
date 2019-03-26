"""Some parts adapted from the TD-VAE code by Xinqiang Ding <xqding@umich.edu>."""

import torch
from torch import nn
from torch.nn import functional as F

from pylego import ops

from ..basetdvae import BaseTDVAE


class DBlock(nn.Module):
    """ A basic building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        t = torch.tanh(self.fc1(input_))
        t = t * torch.sigmoid(self.fc2(input_))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class PreProcess(nn.Module):
    """ The pre-process layer for MNIST image.
    """

    def __init__(self, input_size, processed_x_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input_):
        t = torch.relu(self.fc1(input_))
        t = torch.relu(self.fc2(t))
        return t


class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """

    def __init__(self, z_size, hidden_size, x_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class TDVAE(nn.Module):
    """ The full TD-VAE model with jumpy prediction.
    """

    def __init__(self, x_size, processed_x_size, b_size, z_size, layers=2):
        super().__init__()
        self.layers = layers
        x_size = x_size
        processed_x_size = processed_x_size
        b_size = b_size
        z_size = z_size

        # input pre-process layer
        self.process_x = PreProcess(x_size, processed_x_size)

        # Multilayer LSTM for aggregating belief states
        self.b_rnn = ops.MultilayerLSTM(input_size=processed_x_size, hidden_size=b_size, layers=layers,
                                        every_layer_input=True, use_previous_higher=True)

        # Two layer state model is used. Sampling is done by sampling
        # higher layer first.
        # belief to state (b to z)
        # (this is corresponding to P_B distribution in the reference;
        # weights are shared across time but not across layers.)
        self.z_b = nn.ModuleList([DBlock(b_size + (z_size if layer < layers - 1 else 0), 50, z_size)
                                  for layer in range(layers)])

        # Given belief and state at time t2, infer the state at time t1
        self.z_z_b = nn.ModuleList([DBlock(b_size + layers * z_size + (z_size if layer < layers - 1 else 0), 50, z_size)
                                    for layer in range(layers)])


        # Given the state at time t1, model state at time t2 through state transition
        self.z_z = nn.ModuleList([DBlock(layers * z_size + (z_size if layer < layers - 1 else 0), 50, z_size)
                                  for layer in range(layers)])

        # state to observation
        self.x_z = Decoder(layers * z_size, 200, x_size)

    def forward(self, x, t1, t2):
        # pre-precess image x
        processed_x = self.process_x(x)  # TODO make sure max x length is t2 + 1

        # aggregate the belief b  # XXX should each stochastic layer receive the entire b (all layers)?
        b = self.b_rnn(processed_x)  # size: bs, time, layers, dim
        b1, b2 = b[:, t1], b[:, t2]  # sizes: bs, layers, dim  TODO element-wise indexing, needs batched times

        # q_B(z2 | b2)
        qb_z2_b2_mus, qb_z2_b2_logvars, qb_z2_b2s = [], [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](b2[:, layer])
            else:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](torch.cat([b2[:, layer], qb_z2_b2], dim=1))
            qb_z2_b2_mus.insert(0, qb_z2_b2_mu)
            qb_z2_b2_logvars.insert(0, qb_z2_b2_logvar)

            qb_z2_b2 = ops.reparameterize_gaussian(qb_z2_b2_mu, qb_z2_b2_logvar, self.training)
            qb_z2_b2s.insert(0, qb_z2_b2)

        qb_z2_b2_mu = torch.cat(qb_z2_b2_mus, dim=1)
        qb_z2_b2_logvar = torch.cat(qb_z2_b2_logvars, dim=1)
        qb_z2_b2 = torch.cat(qb_z2_b2s, dim=1)

        # q_S(z1 | z2, b1, b2) ~= q_S(z1 | z2, b1)
        qs_z1_z2_b1_mus, qs_z1_z2_b1_logvars, qs_z1_z2_b1s = [], [], []
        for layer in range(self.layers - 1, -1, -1):  # TODO optionally condition t2 - t1
            if layer == self.layers - 1:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z_z_b(torch.cat([qb_z2_b2, b1[:, layer]], dim=1))
            else:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z_z_b(torch.cat([qb_z2_b2, b1[:, layer], qs_z1_z2_b1], dim=1))
            qs_z1_z2_b1_mus.insert(0, qs_z1_z2_b1_mu)
            qs_z1_z2_b1_logvars.insert(0, qs_z1_z2_b1_logvar)

            qs_z1_z2_b1 = ops.reparameterize_gaussian(qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, self.training)
            qs_z1_z2_b1s.insert(0, qs_z1_z2_b1)

        qs_z1_z2_b1_mu = torch.cat(qs_z1_z2_b1_mus, dim=1)
        qs_z1_z2_b1_logvar = torch.cat(qs_z1_z2_b1_logvars, dim=1)
        qs_z1_z2_b1 = torch.cat(qs_z1_z2_b1s, dim=1)

        # p_T(z2 | z1), also conditions on q_B(z2) from higher layer
        pt_z2_z1_mus, pt_z2_z1_logvars = [], []
        for layer in range(self.layers - 1, -1, -1):  # TODO optionally condition t2 - t1
            if layer == self.layers - 1:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z_z(qs_z1_z2_b1)
            else:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z_z(torch.cat([qs_z1_z2_b1, qb_z2_b2s[layer + 1]], dim=1))
            pt_z2_z1_mus.insert(0, pt_z2_z1_mu)
            pt_z2_z1_logvars.insert(0, pt_z2_z1_logvar)

        pt_z2_z1_mu = torch.cat(pt_z2_z1_mus, dim=1)
        pt_z2_z1_logvar = torch.cat(pt_z2_z1_logvars, dim=1)

        # p_B(z1 | b1)
        pb_z1_b1_mus, pb_z1_b1_logvars = [], []
        for layer in range(self.layers - 1, -1, -1):  # TODO optionally condition t2 - t1
            if layer == self.layers - 1:
                pb_z1_b1_mu, pb_z1_b1_logvar = self.z_b(b1[:, layer])
            else:
                pb_z1_b1_mu, pb_z1_b1_logvar = self.z_b(torch.cat([b1[:, layer], qs_z1_z2_b1s[layer + 1]], dim=1))
            pb_z1_b1_mus.insert(0, pb_z1_b1_mu)
            pb_z1_b1_logvars.insert(0, pb_z1_b1_logvar)

        pb_z1_b1_mu = torch.cat(pb_z1_b1_mus, dim=1)
        pb_z1_b1_logvar = torch.cat(pb_z1_b1_logvars, dim=1)

        # p_D(x2 | z2)
        pd_x2_z2 = self.x_z(qb_z2_b2)

        return (qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar, qb_z2_b2_mu, qb_z2_b2_logvar,
                qb_z2_b2, pt_z2_z1_mu, pt_z2_z1_logvar, pd_x2_z2)

    def rollout(self, x, t1, t2):  # TODO call from visualize
        # pre-precess image x
        processed_x = self.process_x(x)  # TODO make sure max x length is t2 + 1

        # aggregate the belief b  # XXX should each stochastic layer receive the entire b (all layers)?
        b = self.b_rnn(processed_x)[:, t1]  # size: bs, time, layers, dim

        # compute z from b
        p_z_bs = []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                p_z_b_mu, p_z_b_logvar = self.z_b[layer](b[:, layer])
            else:
                p_z_b_mu, p_z_b_logvar = self.z_b[layer](torch.cat([b[:, layer], p_z_b], dim=1))
            p_z_b = ops.reparameterize_gaussian(p_z_b_mu, p_z_b_logvar, False)
            p_z_bs.insert(0, p_z_b)

        z = torch.cat(p_z_bs, dim=1)
        rollout_x = []

        for _ in range(t2 - t1):
            next_z = []
            for layer in range(self.layers - 1, -1, -1):  # TODO optionally condition n
                if layer == self.layers - 1:
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z_z(z)
                else:
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z_z(torch.cat([z, pt_z2_z1], dim=1))
                pt_z2_z1 = ops.reparameterize_gaussian(pt_z2_z1_mu, pt_z2_z1_logvar, False)
                next_z.insert(0, pt_z2_z1)

            z = torch.cat(next_z, dim=1)
            rollout_x.append(self.x_z(z))

        return torch.stack(rollout_x, dim=1)


class TDVAEModel(BaseTDVAE):

    def __init__(self, flags, *args, **kwargs):
        # XXX hardcoded for moving MNIST
        super().__init__(TDVAE(28 * 28, 28 * 28, flags.b_size, flags.z_size, flags.layers), flags, *args, **kwargs)

    def loss_function(self, forward_ret, labels=None):
        x2 = labels
        (qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar, qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2,
         pt_z2_z1_mu, pt_z2_z1_logvar, pd_x2_z2) = forward_ret

        batch_size = x2.size(0)

        kl_div_qs_pb = ops.kl_div_gaussian(qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar).mean()

        sampled_kl_div_qb_pt = (ops.gaussian_log_prob(qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2) -
                                ops.gaussian_log_prob(pt_z2_z1_mu, pt_z2_z1_logvar, qb_z2_b2)).mean()

        bce = F.binary_cross_entropy(pd_x2_z2, x2, reduction='sum') / batch_size
        bce_optimal = F.binary_cross_entropy(x2, x2, reduction='sum') / batch_size
        bce_diff = bce - bce_optimal

        loss = bce_diff + kl_div_qs_pb + sampled_kl_div_qb_pt

        return loss, bce_diff, kl_div_qs_pb, sampled_kl_div_qb_pt, bce_optimal
