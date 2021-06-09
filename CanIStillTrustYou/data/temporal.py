import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

random.seed(0)
holdpct = 0.1
holdmin = 50
#hold_out_size = int(max(holdpct * min(len(data1), len(data2)), holdmin))

def read(fname='./raw/recidivism_1978.csv', ycol='RECID'):
    data = pd.read_csv(fname)
    y = data[ycol]
    X = data.drop(columns=['RECID'])
    return X, y

def clean(X, y, fname='./cleaned/'):
    hold_out_size = int(max(holdpct * len(X), holdmin))
    res = train_test_split(X, y, test_size=hold_out_size, random_state=0)
    res[0].to_csv(fname + '_train_X.csv')
    res[1].to_csv(fname + '_test_X.csv')
    res[2].to_csv(fname + '_train_y.csv')
    res[3].to_csv(fname + '_test_y.csv')

def main():
    X1, y1 = read()
    X2, y2 = read(fname='./raw/recidivism_1980.csv')
    clean(X1, y1, fname='./cleaned/temporal/D1')
    clean(X2, y2, fname='./cleaned/temporal/D2')

if __name__ == '__main__':
    main()

