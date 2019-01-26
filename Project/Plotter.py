import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, train_losses, test_losses):
        self.train_losses = train_losses
        self.test_losses = test_losses

    def plot(self):
        """Plot the loss per epoch"""

        plt.plot(range(len(self.train_losses)), self.train_losses, label='train loss')
        plt.plot(range(len(self.test_losses)), self.test_losses, label='test loss')

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Cross Entropy Loss')
        plt.grid(True)
        plt.show()
