from pylego.model import Model


class BaseTDVAE(Model):

    def __init__(self, model, flags, *args, **kwargs):
        self.flags = flags
        super().__init__(model=model, *args, **kwargs)
