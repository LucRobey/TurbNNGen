import matplotlib.pyplot as plt

def plot_predictions(targets, preds):
    plt.figure(figsize=(10, 7))  # Adjusted figure size for a 2x2 layout
    outputs_nb = targets.shape[1]
    output_labels = {0: 'c1', 1: 'c2', 2: 'L', 3: 'epsilon'}

    for i in range(outputs_nb):
        plt.subplot(2, 2, i+1) 
        plt.scatter(targets[:, i], preds[:, i])
        plt.title(f"Actual vs Predicted values of {output_labels[i]}")
        plt.xlabel(f"Actual {output_labels[i]}")
        plt.ylabel(f"Predicted {output_labels[i]}")

    plt.tight_layout()  
    plt.show()
