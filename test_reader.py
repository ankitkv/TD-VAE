from pylego.misc import save_comparison_grid

from readers.moving_mnist import MovingMNISTReader


if __name__ == '__main__':
    reader = MovingMNISTReader('data/MNIST')
    for i, batch in enumerate(reader.iter_batches('train', 4, max_batches=5)):
        print(batch.size())
        if i < 3:
            batch = batch[0].numpy().reshape(20, 1, 28, 28)
            save_comparison_grid('seq%d.png' % i, batch)
