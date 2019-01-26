import random


class Trainer():
    def __init__(self, model, optimizer, criterion, epochs, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, train_X, train_y, test_X, test_y):

        train_losses = []
        test_losses = []

        for e in range(self.epochs):
            # train_X, train_y = self.shuffle_data(train_X, train_y)

            train_loss = self.forward_batch(train_X, train_y, is_train=True)
            test_loss = self.forward_batch(test_X, test_y, is_train=False)

            print("epoch: {} | train loss: {} | test_loss: {}".format(e + 1, train_loss, test_loss))

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        return train_losses, test_losses

    def forward_batch(self, X, y, is_train):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        num_batches = int(len(X) / self.batch_size)
        epoch_loss = 0.0
        idx = 0

        for j in range(num_batches):

            batch_X = X[idx:idx + self.batch_size * 2]
            batch_y = y[idx:idx + self.batch_size]
            idx += self.batch_size

            if is_train:
                self.optimizer.zero_grad()

            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            if is_train:
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss

        # calculate loss for epoch
        epoch_loss /= (num_batches * self.batch_size)
        return epoch_loss

    # def shuffle_data(self, X, y, seed=4):
    #     c = list(zip(X, y))
    #     random.Random(seed).shuffle(c)
    #     X, y = zip(*c)
    #     return X, y
