import random
import torch


class Trainer():
    def __init__(self, model, optimizer, criterion, epochs, batch_size, to_cuda):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.to_cuda = to_cuda

    def train(self, train_X, train_y, test_X, test_y):

        train_losses = []
        test_losses = []

        if self.to_cuda:
            self.model.cuda()

        for e in range(self.epochs):
            # train_X, train_y = self.shuffle_data(train_X, train_y)

            train_loss, train_acc = self.forward_batch(train_X, train_y, is_train=True)
            test_loss, test_acc = self.forward_batch(test_X, test_y, is_train=False)

            print("epoch: {} | train_loss: {} | train_acc: {} | test_loss: {} | test_acc: {}".format(e + 1, train_loss, train_acc, test_loss, test_acc))

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        return train_losses, test_losses

    def forward_batch(self, X, y, is_train):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        num_batches = int(len(y) / self.batch_size)
        epoch_loss = 0.0
        accuracy = 0.0
        x_idx = 0
        y_idx = 0

        for j in range(num_batches):

            batch_X = X[x_idx:x_idx + self.batch_size * 2]
            batch_y = y[y_idx:y_idx + self.batch_size]
            x_idx += self.batch_size * 2
            y_idx += self.batch_size

            if is_train:
                self.optimizer.zero_grad()

            outputs = self.model(batch_X)

            if self.to_cuda:
                batch_y = torch.Tensor(batch_y).cuda()
            else:
                batch_y = torch.Tensor(batch_y)

            loss = self.criterion(outputs, batch_y)

            if is_train:
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss
            for i in range(len(outputs)):
                if abs(outputs[i] - batch_y[i]) < 0.5:
                    accuracy += 1

        # calculate loss for epoch
        epoch_loss /= (num_batches * self.batch_size)
        accuracy /= (num_batches * self.batch_size)
        return epoch_loss, accuracy

    # def shuffle_data(self, X, y, seed=4):
    #     c = list(zip(X, y))
    #     random.Random(seed).shuffle(c)
    #     X, y = zip(*c)
    #     return X, y
