import torch
import math
import numpy as np


class Trainer():
    def __init__(self, model, optimizer, learning_rate, criterion, epochs, batch_size, is_attn, to_cuda):
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_attn = is_attn
        self.to_cuda = to_cuda

    def train(self, train_X, train_y, test_X, test_y):

        train_losses = []
        test_losses = []
        file = open(
            "Results/output-epochs_{}-learning_rate_{}-batch_size_{}-is_attn_{}.txt".format(self.epochs,
                                                                                            self.learning_rate,
                                                                                            self.batch_size,
                                                                                            self.is_attn, ), "w+")

        if self.to_cuda:
            self.model.cuda()

        for e in range(self.epochs):
            train_loss, train_acc = self.forward_batch(train_X, train_y, is_train=True)
            test_loss, test_acc = self.forward_batch(test_X, test_y, is_train=False)

            print("epoch: {} | train_loss: {} | train_acc: {} | test_loss: {} | test_acc: {}".format(e + 1, train_loss,
                                                                                                     train_acc,
                                                                                                     test_loss,
                                                                                                     test_acc))

            file.write(
                "epoch: {} | train_loss: {} | train_acc: {} | test_loss: {} | test_acc: {}\n".format(e + 1, train_loss,
                                                                                                     train_acc,
                                                                                                     test_loss,
                                                                                                     test_acc))
            train_losses.append(train_loss)
            test_losses.append(test_loss)

        file.close()

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

            if self.is_attn:
                batch_X, lens = self.prepare_batch(batch_X)
                outputs = self.model(batch_X, lens, False)
            else:
                outputs = self.model(batch_X)

            if self.to_cuda:
                batch_y = torch.Tensor(batch_y).cuda()
            else:
                batch_y = torch.Tensor(batch_y)


            loss = self.criterion(outputs, batch_y)


            if is_train:

                loss.backward()
                self.optimizer.step()

            cur_loss = loss.item()
            epoch_loss += cur_loss
            if (math.isnan(cur_loss) or math.isnan(epoch_loss)):
                print(cur_loss)
                print(epoch_loss)

            for i in range(len(outputs)):
                if outputs[i] > 0:
                    if batch_y[i] == 1:
                        accuracy += 1
                else:
                    if batch_y[i] == 0:
                        accuracy += 1


        # calculate loss for epoch
        epoch_loss /= (num_batches * self.batch_size)
        accuracy /= (num_batches * self.batch_size)
        return epoch_loss, accuracy

    def prepare_batch(self, batch_x):
        # get the length of each sentence  
        batch_lengths = [len(sentence) for sentence in batch_x]
        batch_x = [self.model.idx_vecs(sent) for sent in batch_x]

        # create an empty matrix with padding tokens
        pad_token = 0
        longest_sent = max(batch_lengths)
        batch_size = len(batch_x)
        padded_batch = np.ones((batch_size, longest_sent)) * pad_token

        # copy over the actual sequences
        for i, sent_len in enumerate(batch_lengths):
            sequence = batch_x[i]
            padded_batch[i, 0:sent_len] = sequence.cpu()[:sent_len]

        if self.to_cuda:
            batch = [torch.cuda.LongTensor(l) for l in padded_batch]
        else:
            batch = [torch.LongTensor(l) for l in padded_batch]

        return torch.stack(batch), batch_lengths