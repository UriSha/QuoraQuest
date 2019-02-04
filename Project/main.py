import torch
import torch.nn as nn
from DataInit import DataInit
from WordVecs import WordVecs
from Models.SimpleModel import SimpleModel
from Models.AttnModel import AttnModel
from Trainer import Trainer
# from Plotter import Plotter
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--emdpath',
                        help="where the embeddings file is located",
                        default='Data/fasttext.vec')
    parser.add_argument('-dp', '--datapath',
                        help="where the data file is located",
                        default='Data/train.csv')
    parser.add_argument('-model', '--model',
                        help="which model to use",
                        default='simple')
    parser.add_argument('-lr', '--lr',
                        help="learning rate to use",
                        default=0.001)
    parser.add_argument('-batch_size', '--batch_size',
                        help="size of batch",
                        default=128)
    parser.add_argument('-epochs', '--epochs',
                        help="number of epochs",
                        default=250)
    parser.add_argument('-test_ratio', '--test_ratio',
                        help="test/train ratio",
                        default=0.1)
    parser.add_argument('-cuda', '--cuda',
                        help="use cuda",
                        default=True)
    args = parser.parse_args()

    # Consts
    print("Init Consts")
    data_file_path = args.datapath
    emmbedings_file_path = args.emdpath
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    test_ratio = args.test_ratio
    cuda = args.cuda

    print("Start DataInit")
    data_init = DataInit(data_file_path)

    train_X, test_X, train_y, test_y = data_init.get_train_test_data(test_ratio)

    print("Importing embeddings")
    src_vecs = WordVecs(emmbedings_file_path)

    if args.model == 'simple':
        print("Init simple Model")
        is_attn = False
        model = SimpleModel(src_vecs, cuda)
    else:
        print("Init attention Model")
        is_attn = True
        model = AttnModel(src_vecs, batch_size, cuda)

    print("Model parametrs ", model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduce=True, size_average=False)

    print("Init Trainer")
    trainer = Trainer(model, optimizer,learning_rate, criterion, epochs, batch_size, is_attn, cuda)

    print("Start training with parameters:")
    print("batch_size: {} | epochs: {} | learning_rate: {} | train_ratio: {}".format(batch_size, epochs, learning_rate,
                                                                                     test_ratio))
    train_losses, test_losses = trainer.train(train_X, train_y, test_X, test_y)

    # plotter = Plotter(train_losses, test_losses)
    # plotter.plot()


if __name__ == '__main__':
    main()
