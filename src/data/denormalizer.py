# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

scaler = MinMaxScaler()
import sys

arguments = sys.argv

#=================================================POST-PROCESSING
"""
filename         : chemin pour le fichier des résultats
scaler_save_name : nom du fichier contenant la mémoire du scaler

"""
def denormalize(filename , scaler_save_name):
    #We load the scaler and the results from NN
    print("\nOn essaie d'ouvir le fichier :"+filename)
    Results = dict(np.load(filename))
    print("succès")
    print("\nOn essaie de load :"+scaler_save_name)
    scaler = joblib.load(scaler_save_name)
    print("succès")
    
    #On dénormalise
    
    print("\nOn charge Y")
    normalized_Y=Results["Y"]
    print(normalized_Y)
    
    #We modify the file
    print("\nOn dénormalise")
    denormalised_Y = scaler.inverse_transform(normalized_Y)
    print(denormalised_Y)
    print("\nOn modifie le fichier")
    Results["Y"]=denormalised_Y
    print("\nOn save les modif")
    np.savez(filename,**Results)
    print("succès")
    
denormalize(arguments[1],arguments[2])