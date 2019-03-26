import torch

from pylego.model import Model


class BaseTDVAE(Model):

    def __init__(self, model, flags, *args, **kwargs):
        self.flags = flags
        super().__init__(model=model, *args, **kwargs)

    def prepare_batch(self, data):
        batch, t1, t2 = data
        batch = batch[:, :t2.max() + 1]
        batch, t1, t2 = super().prepare_batch([batch, t1, t2])
        x2 = torch.gather(batch, 1, t2[:, None, None].expand(-1, -1, batch.size(2))).squeeze(1)
        return batch, t1, t2, x2
