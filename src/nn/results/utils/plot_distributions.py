import matplotlib.pyplot as plt
import numpy as np

def plot_distributions(targets, preds, labels):
    plt.figure(figsize=(10, 7))  # Adjusted figure size for a 2x2 layout
    # outputs_nb = targets.shape[1]
    # output_labels = {0: 'c1', 1: 'c2', 2: 'L', 3: 'epsilon'}
    # output_labels = {i: label for i, label in enumerate(labels)}

    # for i in range(outputs_nb):
    for i, label in enumerate(labels):
        box_x = np.unique(targets[:, i])
        box_y = [preds[:, i][targets[:, i] == x] for x in box_x]
        
        plt.subplot(2, 2, i+1)

        plt.boxplot(box_y)
        plt.violinplot(box_y)
        plt.xticks(np.arange(1,len(box_x)+1), box_x)
        
        plt.title(f"Actual vs Predicted values of {label}")
        plt.xlabel(f"Actual {label}")
        plt.ylabel(f"Predicted {label}")
        
        plt.grid() 

    plt.tight_layout()  
    plt.show()
