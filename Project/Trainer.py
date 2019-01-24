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
            self.forward_batch(train_X, train_y, train_losses, is_train=True)
            self.forward_batch(test_X, test_y, test_losses, is_train=False)

            print("epoch: {} | train loss: {} | test_loss: {}".format(e + 1, train_losses[-1], test_losses[-1]))

        return train_losses, test_losses

    def forward_batch(self, X, y, losses_list, is_train):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        num_batches = int(len(X) / self.batch_size)

        epoch_loss = 0.0
        idx = 0

        for j in range(num_batches):
            batch_X = X[idx:idx + self.batch_size]
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
        losses_list.append(epoch_loss)
