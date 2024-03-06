import pandas as pd
import matplotlib.pyplot as plt

def plot_correlation(targets, preds, labels):
    fig = plt.figure(figsize=(10, 7)) 

    targets_corr = pd.DataFrame(targets).corr()
    preds_corr = pd.DataFrame(preds).corr()

    gs = fig.add_gridspec(2, 3, width_ratios=[10, 10, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])

    fig.suptitle("Correlation matrix")

    p = ax1.imshow(targets_corr, vmin=-1, vmax=1, interpolation='None', aspect='equal')
    ax1.set_xticks(list(range(len(labels))))
    ax1.set_xticklabels(labels)
    ax1.set_yticks(list(range(len(labels))))
    ax1.set_yticklabels(labels)
    ax1.set_title("Ground Truth")

    ax2.imshow(preds_corr, vmin=-1, vmax=1)
    ax2.set_xticks(list(range(len(labels))))
    ax2.set_xticklabels(labels)
    ax2.set_yticks(list(range(len(labels))))
    ax2.set_yticklabels(labels)
    ax2.set_title("Predictions")

    # Create a single color bar with adjusted aspect
    cbar = plt.colorbar(p, cax=ax_cb, orientation='vertical')
    cbar.set_label('Index')

    plt.tight_layout()
    plt.show()
