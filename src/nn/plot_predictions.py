import matplotlib.pyplot as plt

def plot_predictions(targets, preds, labels):
    plt.figure(figsize=(10, 7))  # Adjusted figure size for a 2x2 layout
    # outputs_nb = targets.shape[1]
    # output_labels = {0: 'c1', 1: 'c2', 2: 'L', 3: 'epsilon'}
    # output_labels = {i: label for i, label in enumerate(labels)}

    # for i in range(outputs_nb):
    for i, label in enumerate(labels):
        plt.subplot(2, 2, i+1) 
        plt.scatter(targets[:, i], preds[:, i])
        plt.title(f"Actual vs Predicted values of {label}")
        plt.xlabel(f"Actual {label}")
        plt.ylabel(f"Predicted {label}")

    plt.tight_layout()  
    plt.show()
