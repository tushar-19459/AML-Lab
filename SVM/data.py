import numpy as np
from sklearn.datasets import make_blobs

def LinearSeparableData():
    X,y = make_blobs(n_samples=100,centers=2,random_state=42,cluster_std=1.0)
    y = np.where(y==0,-1,1)
    return X,y

def OverLappingData():
    x_overlap,y_overlap = make_blobs(n_samples=100,centers=2,random_state=42,cluster_std=3)
    y_overlap=np.where(y_overlap==0,-1,1)
    return x_overlap,y_overlap
