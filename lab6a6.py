#importing  necessary packages 
import numpy as np 
import pandas as pd 
from graphviz import Digraph

# building decision tree 
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Feature used for splitting
        self.threshold = threshold  # Threshold for splitting
        self.left = left            # Left child
        self.right = right          # Right child
        self.value = value          # Class label if leaf node

#entropy
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

#information gain 
def information_gain(y, left_y, right_y):
    weight_left = len(left_y) / len(y)
    weight_right = len(right_y) / len(y)
    return entropy(y) - (weight_left * entropy(left_y) + weight_right * entropy(right_y))

#Feature bining 
def bin_all_numeric(df, num_bins=5, bin_type='width', target_col=None):
    df_binned = df.copy()
    for col in df.columns:
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

#Best split 
def best_split(X, y):
    best_ig = -1
    split_feature = None
    split_threshold = None
    
    for feature in X.columns:
        values = np.unique(X[feature])
        for val in values:
            left_idx = X[feature] <= val
            right_idx = X[feature] > val
            if sum(left_idx) == 0 or sum(right_idx) == 0:
                continue
            ig = information_gain(y, y[left_idx], y[right_idx])
            if ig > best_ig:
                best_ig = ig
                split_feature = feature
                split_threshold = val
    return split_feature, split_threshold

#building tree recursively 
def build_tree(X, y, max_depth=None, depth=0):
    if len(np.unique(y)) == 1:
        return DecisionNode(value=y.iloc[0])
    
    if max_depth is not None and depth >= max_depth:
        values, counts = np.unique(y, return_counts=True)
        return DecisionNode(value=values[np.argmax(counts)])
    
    feature, threshold = best_split(X, y)
    if feature is None:
        values, counts = np.unique(y, return_counts=True)
        return DecisionNode(value=values[np.argmax(counts)])
    
    left_idx = X[feature] <= threshold
    right_idx = X[feature] > threshold
    
    left_subtree = build_tree(X[left_idx], y[left_idx], max_depth, depth+1)
    right_subtree = build_tree(X[right_idx], y[right_idx], max_depth, depth+1)
    
    return DecisionNode(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

#prediction function 
def predict_single(node, sample):
    if node.value is not None:
        return node.value
    if sample[node.feature] <= node.threshold:
        return predict_single(node.left, sample)
    else:
        return predict_single(node.right, sample)

def predict(tree, X):
    return X.apply(lambda row: predict_single(tree, row), axis=1)

#visualizing graph 
def visualize_tree(node, feature_names, dot=None, node_id=0):
    if dot is None:
        dot = Digraph()
    
    current_id = str(node_id)
    
    if node.value is not None:
        dot.node(current_id, f"Class: {node.value}", shape='box', style='filled', color='lightgrey')
    else:
        label = f"{node.feature} <= {node.threshold}"
        dot.node(current_id, label)
        left_id = str(2*node_id + 1)
        right_id = str(2*node_id + 2)
        
        visualize_tree(node.left, feature_names, dot, 2*node_id + 1)
        visualize_tree(node.right, feature_names, dot, 2*node_id + 2)
        
        dot.edge(current_id, left_id, label="Yes")
        dot.edge(current_id, right_id, label="No")
    
    return dot

# Load dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# last column is target
target_col = df.columns[-1]

# Bin numeric features
df_binned = bin_all_numeric(df, num_bins=5, bin_type='width', target_col=target_col)

X = df_binned.drop(columns=[target_col])
y = df_binned[target_col]

tree = build_tree(X, y, max_depth=5)
y_pred = predict(tree, X)

# Check accuracy
accuracy = np.sum(y_pred == y) / len(y)
print("Training accuracy:", accuracy)

#decision tree visualization 
dot = visualize_tree(tree, X.columns)
dot.render('decision_tree', format='png', cleanup=True) 
