import matplotlib.pyplot as plt

def plot_losses(n_epochs, losses, legends):
    epochs = range(n_epochs)
    for i in range(len(losses)):
        curr_losses = losses[i]
        curr_legend = legends[i]
        plt.plot(epochs, curr_losses, label=curr_legend)

    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


