import torch
import torch.nn as nn
from DataInit import DataInit
from WordVecs import WordVecs
from Models.SimpleModel import SimpleModel
from Trainer import Trainer
from Plotter import Plotter


def main():

    # Consts
    data_file_path = ""
    emmbedings_file_path = ""
    batch_size = 16
    epochs = 100
    learning_rate = 0.001
    train_ratio = 0.1

    data_init = DataInit(data_file_path)

    train_X, test_X, train_y, test_y = data_init.get_train_test_data(train_ratio)

    src_vecs = WordVecs(emmbedings_file_path)

    model = SimpleModel(src_vecs)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    trainer = Trainer(model, optimizer, criterion, epochs, batch_size)

    train_losses, test_losses = trainer.train(train_X, train_y, test_X, test_y)

    plotter = Plotter(train_losses, test_losses)
    plotter.plot()


if __name__ == '__main__':
    main()
