import matplotlib.pyplot as plt
import numpy as np

def analyse_predictions(targets, preds):
    outputs_nb = (targets.shape)[1]
    output_labels = {0: 'c1', 1: 'c2', 2: 'L', 3: 'epsilon'}
    
    for i in range(0, outputs_nb):
        unique_values, counts = np.unique(targets[:,i], return_counts=True)
        print(f"True values of {output_labels[i]} and their count : ")
        for value, count in zip(unique_values, counts):
            print(f"{value}: {count} occurrences")

        print("")
        unique_values, counts = np.unique(preds[:,i], return_counts=True)
        print(f"Predicted values of {output_labels[i]} and their count: ")
        for value, count in zip(unique_values, counts):
            print(f"{value}: {count} occurrences")
        
        print("")
        print("-------------------------------------------------------------------------")
        