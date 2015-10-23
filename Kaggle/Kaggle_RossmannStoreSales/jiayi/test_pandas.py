# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.read_csv("../data/train.csv")
data_train=pd.read_csv("../data/train.csv")
#dataset=np.genfromtxt(open("../data/train.csv","r"),delimiter=',',dtype='f8')
#data_train.apply(np.isnan)
#data_train[data_train.isnull().any(axis=1)]
data_store=pd.read_csv("../data/store.csv")
#data_store[data_store.isnull().any(axis=1)]
