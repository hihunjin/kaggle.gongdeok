import numpy as np
import pandas as pd

from os.path import basename
def load_data(path="C:/Users/win10/Workspace/kaggle.dataset/statoil-iceberg-classifier-challenge/train_nona.json"):

    
    data = pd.read_json(path)
    data['inc_angle'] = pd.to_numeric(data['inc_angle'],errors='coerce')

    return data

def load_test(path="C:/Users/win10/Workspace/kaggle.dataset/statoil-iceberg-classifier-challenge/test.json"):

    data = pd.read_json(path)
    data['inc_angle'] = pd.to_numeric(data['inc_angle'],errors='coerce')

    return data

def to_numpy(data, with_angle=False, for_train=True):

    band_1 = data['band_1'].apply(lambda x: np.reshape(x, newshape=(75,75)))
    band_2 = data['band_2'].apply(lambda x: np.reshape(x, newshape=(75,75)))
    
    X = np.array([
        np.transpose(np.array([b1,b2]),[1,2,0]) 
        for b1, b2 in zip(band_1, band_2)], dtype=np.float32)

    if for_train:
        if with_angle:
            X1, X2 = X, data['inc_angle'].values
            Y = np.array(data['is_iceberg'], dtype=np.float32)
            return X1, X2, Y 

        else:
            Y = np.array(data['is_iceberg'], dtype=np.float32)
            return X, Y 

    else:
        if with_angle:
            return X, data['inc_angle'].values
        else:
            return X
        



