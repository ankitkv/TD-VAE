import collections

import numpy as np

from pylego import misc

from models.basetdvae import BaseTDVAE
from .basemnist import MovingMNISTBaseRunner


class TDVAERunner(MovingMNISTBaseRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseTDVAE, ['loss', 'bce_diff', 'kl_div_qs_pb', 'sampled_kl_div_qb_pt'])

    def run_batch(self, batch, train=False):
        batch = self.model.prepare_batch(batch)
        loss, bce_diff, kl_div_qs_pb, sampled_kl_div_qb_pt, bce_optimal = self.model.run_loss(batch)
        if train:
            self.model.train(loss, clip_grad_norm=self.flags.grad_norm)

        return collections.OrderedDict([('loss', loss.item()),
                                        ('bce_diff', bce_diff.item()),
                                        ('kl_div_qs_pb', kl_div_qs_pb.item()),
                                        ('sampled_kl_div_qb_pt', sampled_kl_div_qb_pt.item()),
                                        ('bce_optimal', bce_optimal.item())])

    def _visualize_split(self, split, t, n):
        bs = min(self.batch_size, 16)
        batch = next(self.reader.iter_batches(split, bs, shuffle=True, partial_batching=True, threads=self.threads,
                                              max_batches=1))
        batch = self.model.prepare_batch(batch[:, :t + 1])
        out = self.model.run_batch([batch, t, n], visualize=True)

        batch = batch.cpu().numpy()
        out = out.cpu().numpy()
        vis_data = np.concatenate([batch, out], axis=1)
        bs, seq_len = vis_data.shape[:2]
        return vis_data.reshape([bs * seq_len, 1, 28, 28]), seq_len / bs

    def post_epoch_visualize(self, epoch, split):
        if split != 'train':
            print('* Visualizing', split)
            vis_data, aspect = self._visualize_split(split, 10, 5)
            if split == 'test':
                fname = self.flags.log_dir + '/test.png'
            else:
                fname = self.flags.log_dir + '/val%03d.png' % epoch
            misc.save_comparison_grid(fname, vis_data, desired_aspect=aspect, border_shade=1.0)
            print('* Visualizations saved to', fname)

        if split == 'test':
            print('* Generating more visualizations for', split)
            vis_data, aspect = self._visualize_split(split, 0, 15)
            fname = self.flags.log_dir + '/test_more.png'
            misc.save_comparison_grid(fname, vis_data, desired_aspect=aspect, border_shade=1.0)
            print('* More visualizations saved to', fname)
