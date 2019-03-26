import collections

import numpy as np

from pylego import misc

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

    def post_epoch_visualize(self, epoch, split):
        if split != 'train':
            print('* Visualizing', split)
            t, n = 10, 5
            bs = min(self.batch_size, 16)
            batch = next(self.reader.iter_batches(split, bs, shuffle=True, partial_batching=True, threads=self.threads,
                                                  max_batches=1))
            batch = batch[:, :t + 1]
            data = self.model.prepare_batch([batch, t, n], visualize=True)

            out = self.model.run_batch(data, visualize=True)

            batch = batch.numpy()
            out = out.cpu().numpy()
            vis_data = np.concatenate([batch, out], axis=1)
            bs, seq_len = vis_data.shape[:2]
            vis_data = vis_data.reshape([bs * seq_len, 1, 28, 28])

            if split == 'test':
                fname = self.flags.log_dir + '/test.png'
            else:
                fname = self.flags.log_dir + '/val%03d.png' % epoch
            misc.save_comparison_grid(fname, vis_data, desired_aspect=seq_len/bs, border_shade=1.0)
            print('* Visualizations saved to', fname)
