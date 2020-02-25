# import the required libraries
import numpy as np 
import matplotlip.pyplot as plt
import pandas an pd
import sklearn 
from sklearn.datasets import load_boston

#import the data set 
boston = laod_boston()

bos = pd.DataFrame(boston.Data)

print(bos.head(10))
