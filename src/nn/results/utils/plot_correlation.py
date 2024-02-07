import pandas as pd
import matplotlib.pyplot as plt

def plot_correlation(targets, preds, labels):
    plt.figure(figsize=(10, 7)) 

    targets_corr = pd.DataFrame(targets).corr()
    preds_corr = pd.DataFrame(preds).corr()

    fig, ax = plt.subplots(1,2)
    fig.suptitle("Correlation matrix")

    ax[0].matshow(targets_corr, vmin=0, vmax=1)
    ax[0].set_xticks(list(range(len(labels))))
    ax[0].set_xticklabels(labels)
    ax[0].set_yticks(list(range(len(labels))))
    ax[0].set_yticklabels(labels)
    ax[0].set_title("Ground Truth")

    ax[1].matshow(preds_corr, vmin=-1, vmax=1)
    ax[1].set_xticks(list(range(len(labels))))
    ax[1].set_xticklabels(labels)
    ax[1].set_yticks(list(range(len(labels))))
    ax[1].set_yticklabels(labels)
    ax[1].set_title("Predictions")

    plt.tight_layout()
    plt.show()
