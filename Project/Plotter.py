import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, losses):
        self.losses = losses

    def plot(self):
        """Plot the loss per epoch"""
        epoch_count = len(self.losses)
        plt.plot(range(epoch_count), self.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Cross Entropy Loss')
        plt.grid(True)
        plt.show()
