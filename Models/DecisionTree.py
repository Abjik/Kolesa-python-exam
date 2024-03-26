import numpy as np
import pandas as pd

def mse(y):
    n = len(y)
    y_mean = y.mean()
    return ((y-y_mean)**2).mean()

def msep(y, X, Q):
    n = len(y)
    y_right = y.loc[X[X>Q].index]
    y_left = y.loc[X[X<=Q].index]

    mse_left = mse(y_left)
    mse_right = mse(y_right)
    
    n_left = len(y_left)
    n_right = len(y_right)
    
    return mse(y) - (n_left*mse_left/n + n_right*mse_right/n) 

class TreeNode:
    def __init__(self, split_column, split_value, predicted_classes):
        self.split_column = split_column
        self.split_value = split_value
        self.predicted_classes = predicted_classes
        self.left = None
        self.right = None
        
class TreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs if max_leafs > 1 else 2
        self.bins = bins
        self.bin_values = None  # Initialize bin_values attribute
    
    def fit(self,X, y):
        self.leafs_cnt = 1

        # Calculate bins if bins is not None
        if self.bins is not None:
            self.bin_values = {}  # Store calculated bin values for each column
            for column in X.columns:
                unique_values = np.unique(X[column])
                if len(unique_values) <= self.bins - 1:
                    self.bin_values[column] = unique_values
                else:
                    hist, bins = np.histogram(X[column], bins=self.bins)
                    self.bin_values[column] = bins

        self.node = self.build_tree(X, y, 0)
        
    def predict(self, X):
        self.y_predict = pd.Series(index=X.index)
        self.prediction(self.node, X)
        return self.y_predict
    
    def prediction(self, node, X):
        if type(node) == np.float64:
            self.y_predict[X.index] = node
            return
        
        left_indexes = X[node.split_column] <= node.split_value
        right_indexes = X[node.split_column] > node.split_value
        self.prediction(node.left, X[left_indexes])
        self.prediction(node.right, X[right_indexes])     
        
    def build_tree(self, X, y, current_depth):
        if self.leafs_cnt >= self.max_leafs:
            return self.build_leaf(X,y)
        
        if X.shape[0] <= 1 or len(np.unique(y)) <= 1:
            return self.build_leaf(X,y)
        
        split_column, Q, ig = self.get_best_split(X, y)
        node = TreeNode(split_column, Q, -1)
        
        
        if current_depth < self.max_depth and len(y) >= self.min_samples_split:

            left_indices = X[split_column] <= Q
            right_indices = X[split_column] > Q

            X_left = X[left_indices]
            y_left = y[left_indices]
            X_right = X[right_indices]
            y_right = y[right_indices]
    
            self.leafs_cnt += 1
            node.left = self.build_tree(X_left, y_left, current_depth + 1)
            node.right = self.build_tree(X_right, y_right, current_depth + 1)           
            
        else:
            return self.build_leaf(X,y)
        return node
    
    def build_leaf(self, X, y):
        return np.sum(y[X.index]) / len(y[X.index])        
    
    def get_best_split(self, X, y):
        best_ig = 0
        best_Q = 0
        best_column_name = None
        prev_value = None
       
        for column in X.columns:
            if self.bins is None:
                unique_values = np.unique(X[column])
                if len(unique_values) <= 1:
                    continue
                for value in unique_values:
                    ig = msep(y, X[column], value)
                    if ig > best_ig:
                        best_ig = ig
                        best_Q = value
                        best_column_name = column
            else:
                for value in self.bin_values[column]:
                    ig = msep(y, X[column], value)
                    if ig > best_ig:
                        best_ig = ig
                        best_Q = value
                        best_column_name = column
                          
        return best_column_name, best_Q, best_ig  
    
    def print_tree(self, node):
        if type(node) == np.float64:
            print(node)
            return
        print(node.split_column, node.split_value)
        self.print_tree(node.left)
        self.print_tree(node.right)