import torch
import torch.nn as nn
from DataInit import DataInit
from WordVecs import WordVecs
from Models.SimpleModel import SimpleModel
from Trainer import Trainer
from Plotter import Plotter
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--emdpath',
                        help="where to dump weights during training (default: ./models)",
                        default='Data/fasttext.vec')
    parser.add_argument('-dp', '--datapath',
                        help="where to dump weights during training (default: ./models)",
                        default='Data/train.csx.csv')
    args = parser.parse_args()

    # Consts
    data_file_path = args.datapath
    emmbedings_file_path = args.emdpath
    batch_size = 256
    epochs = 200
    learning_rate = 0.001
    train_ratio = 0.1
    cuda = True

    data_init = DataInit(data_file_path)

    train_X, test_X, train_y, test_y = data_init.get_train_test_data(train_ratio)

    src_vecs = WordVecs(emmbedings_file_path)

    model = SimpleModel(src_vecs, cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    trainer = Trainer(model, optimizer, criterion, epochs, batch_size, cuda)

    train_losses, test_losses = trainer.train(train_X, train_y, test_X, test_y)

    plotter = Plotter(train_losses, test_losses)
    plotter.plot()


if __name__ == '__main__':
    main()
