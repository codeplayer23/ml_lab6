#importing the necesary packages 
import pandas as pd 
import numpy as np 

#importing the dataset 
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

#extracting features 
X = df.iloc[:,0:196].where(df["LABEL"] == 2).dropna()
Y = df["LABEL"].where(df["LABEL"] == 2).dropna()

#calculating entropy 
counts = df["LABEL"].value_counts().values
probability = counts / counts.sum()
entropy = -np.sum(probability * np.log2(probability))

print("Entropy :",entropy)