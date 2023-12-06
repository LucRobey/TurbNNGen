# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys

arguments = sys.argv

scaler = MinMaxScaler()

#=================================================PRE-PROCESSING
"""
filename         : chemin pour le fichier des données à normaliser
scaler_save_name : nom du fichier contenant la mémoire du scaler

"""
def normalize(filename , scaler_save_name):
    
    
    print("\nOn essaie d'ouvir le fichier :"+filename)
    DATA = dict(np.load(filename))
    print("succès")
    print("\nOn charge Y")
    Y=DATA["Y"]
    print(Y)
    #we normalize
    print("\nOn normalise")
    normalized_Y = scaler.fit_transform(Y)
    print(normalized_Y)
    #we modify the file
    print("\nOn modifie le fichier")
    DATA["Y"]=normalized_Y
    print("\nOn save les modif")
    np.savez(filename,**DATA)
    print("succès")
    
    #we save the scaler
    print("\nOn save le scaler dans :" + scaler_save_name)
    joblib.dump(scaler,scaler_save_name)
    
    


normalize(arguments[1] , arguments[2])