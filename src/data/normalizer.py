# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

#=================================================PRE-PROCESSING
"""
filename         : chemin pour le fichier des données à normaliser
scaler_save_name : nom du fichier contenant la mémoire du scaler
"""
def normalize(filename, scalerpath, save_scaler=False):
    scaler = MinMaxScaler()
    if not save_scaler:
        scaler = joblib.load(scalerpath) 
    DATA = dict(np.load(filename))
    Y=DATA["Y"]
    normalized_Y = scaler.fit_transform(Y)
    DATA["Y"]=normalized_Y
    np.savez(filename,**DATA)
    if save_scaler:
        joblib.dump(scaler, scalerpath)
    
