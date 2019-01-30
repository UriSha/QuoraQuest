import torch
import torch.nn as nn
from DataInit import DataInit
from WordVecs import WordVecs
from Models.SimpleModel import SimpleModel
from Models.AttnModel import AttnModel
from Trainer import Trainer
#from Plotter import Plotter
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
    args = parser.parse_args()

    # Consts
    print("Init Consts")
    data_file_path = args.datapath
    emmbedings_file_path = args.emdpath
    batch_size = 256
    epochs = 200
    learning_rate = 0.001
    train_ratio = 0.1
    cuda = True

    print("Start DataInit")
    data_init = DataInit(data_file_path)

    train_X, test_X, train_y, test_y = data_init.get_train_test_data(train_ratio)

    print("Importing embeddings")
    src_vecs = WordVecs(emmbedings_file_path)

    if args.model == 'simple':
        print("Init simple Model")
        is_attn = False
        model = SimpleModel(src_vecs, cuda)
    else:
        print("Init attention Model")
        is_attn = True
        model = AttnModel(src_vecs, cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("Init Trainer")
    trainer = Trainer(model, optimizer, criterion, epochs, batch_size, is_attn, cuda)

    print("Start training")
    train_losses, test_losses = trainer.train(train_X, train_y, test_X, test_y)

    #plotter = Plotter(train_losses, test_losses)
    #plotter.plot()


if __name__ == '__main__':
    main()
