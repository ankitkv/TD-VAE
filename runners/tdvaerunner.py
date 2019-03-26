import collections

import numpy as np

from models.basetdvae import BaseTDVAE
from .basemnist import MovingMNISTBaseRunner


class TDVAERunner(MovingMNISTBaseRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseTDVAE, ['loss', 'bce_diff', 'kl_div_qs_pb', 'sampled_kl_div_qb_pt'])

    def run_batch(self, batch, train=False):
        t1 = np.random.randint(self.flags.seq_len - self.flags.t_diff_max, size=batch.shape[0])
        t2 = t1 + np.random.randint(self.flags.t_diff_min, self.flags.t_diff_max + 1, size=batch.shape[0])
        batch, t1, t2, x2 = self.model.prepare_batch([batch, t1, t2])
        loss, bce_diff, kl_div_qs_pb, sampled_kl_div_qb_pt, bce_optimal = self.model.run_loss([batch, t1, t2],
                                                                                              labels=x2)
        if train:
            self.model.train(loss, clip_grad_norm=self.flags.grad_norm)

        return collections.OrderedDict([('loss', loss.item()),
                                        ('bce_diff', bce_diff.item()),
                                        ('kl_div_qs_pb', kl_div_qs_pb.item()),
                                        ('sampled_kl_div_qb_pt', sampled_kl_div_qb_pt.item()),
                                        ('bce_optimal', bce_optimal.item())])

    def post_epoch_visualize(self, epoch, split):  # TODO
        if split != 'train':
            print('* Visualizing', split)
            print('* Visualizations saved.')
