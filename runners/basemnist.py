import importlib

from readers.moving_mnist import MovingMNISTReader
from pylego import misc, runner


class MovingMNISTBaseRunner(runner.Runner):

    def __init__(self, flags, model_class, log_keys, *args, **kwargs):
        self.flags = flags
        reader = MovingMNISTReader(flags.data_path, seq_len=flags.seq_len)
        summary_dir = flags.log_dir + '/summary'
        super().__init__(reader, flags.batch_size, flags.epochs, summary_dir, log_keys=log_keys,
                         threads=flags.threads, print_every=flags.print_every, visualize_every=flags.visualize_every,
                         *args, **kwargs)
        model_class = misc.get_subclass(importlib.import_module('models.' + self.flags.model), model_class)
        self.model = model_class(self.flags, optimizer=flags.optimizer, learning_rate=flags.learning_rate,
                                 cuda=flags.cuda, load_file=flags.load_file, save_every=flags.save_every,
                                 save_file=flags.save_file, debug=flags.debug)
