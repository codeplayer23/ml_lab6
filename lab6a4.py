#importing the packages 
import pandas as pd 
import numpy as np 

def bin_all_numeric(df, num_bins=5, bin_type='width', target_col=None):
    
    df_binned = df.copy()
    
    for col in df.columns:
        # Skip target column if specified
        if col == target_col:
            continue
        
        if df[col].dtype in [np.int64, np.float64]:
            if bin_type == 'width':
                df_binned[col] = pd.cut(df[col], bins=num_bins, labels=False)
            elif bin_type == 'frequency':
                df_binned[col] = pd.qcut(df[col], q=num_bins, labels=False, duplicates='drop')
            else:
                raise ValueError("bin_type must be 'width' or 'frequency'")
    
    return df_binned


#Finding Entropy 
def entropy(target_col):
    values, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / counts.sum()
    ent = -np.sum(probabilities * np.log2(probabilities)) 
    return ent

#Information Gain 
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset = data[data[feature] == v]
        weighted_entropy += (c / len(data)) * entropy(subset[target])
    ig = total_entropy - weighted_entropy
    return ig

#Best Root Feature 
def best_root_feature(data, target):
    features = [col for col in data.columns if col != target]
    ig_values = {}
    
    for feature in features:
        ig_values[feature] = information_gain(data, feature, target)
    
    root_feature = max(ig_values, key=ig_values.get)
    return root_feature, ig_values[root_feature]

#Loading dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

#last column is target
target_col = df.columns[-1]

# Bin all numeric features
df_binned = bin_all_numeric(df, num_bins=5, bin_type='width', target_col=target_col)
print(df_binned.head())

#Detect root node
root, ig = best_root_feature(df_binned, target_col)
print(f"Root Feature: {root} with Information Gain: {ig}")
