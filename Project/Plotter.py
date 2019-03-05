import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, first_losses, fist_label, sec_losses, sec_label):
        self.first_losses = first_losses
        self.fist_label = fist_label
        self.sec_losses = sec_losses
        self.sec_label = sec_label

    def plot(self, file_name):
        """Plot the loss per epoch"""

        line1, = plt.plot(range(len(self.first_losses)), self.first_losses, label=self.fist_label)

        line2, = plt.plot(range(len(self.sec_losses)), self.sec_losses, label=self.sec_label)
        plt.legend(handles=[line1,line2])
        plt.xlabel('epoch')
        # plt.ylabel('accuracy')
        # plt.title('Accuracy per epoch')
        plt.ylabel('loss')
        plt.title('Cross entropy loss')
        plt.grid(True)
        plt.savefig(file_name)
