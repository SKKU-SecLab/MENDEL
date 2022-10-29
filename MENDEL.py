import pandas as pd
import numpy as np
import sys
import pickle as pkl
import time
import math
import random

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from scipy.special import rel_entr
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
import tensorflow as tf
import seaborn as sns
from scipy.special import kl_div
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler, RobustScaler
import xgboost as XGB


### datasets
# swat

with open('./swat/SWaT_train.pkl', 'rb') as f:
    train_swat_a = pkl.load(f)
with open('./swat/SWaT_test.pkl', 'rb') as f:
    test_swat_a = pkl.load(f)
with open('./swat/SWaT_test_label.pkl', 'rb') as f:
    label_swat_a = pkl.load(f)

with open("./wadi/WADI_A1_train.pkl", "rb") as f:
    train_wadi = pkl.load(f)

with open("./wadi/WADI_A1_test.pkl", "rb") as f:
    test_wadi = pkl.load(f)

with open("./wadi/WADI_A1_test_label.pkl", "rb") as f:
    label_wadi = pkl.load(f)

### dataset 0-1 scaling
scaler = MinMaxScaler()

swat_train_df = scaler.fit_transform(train_swat_a)
swat_train_df = pd.DataFrame(swat_train_df)

swat_test_df = scaler.fit_transform(test_swat_a)
swat_test_df = pd.DataFrame(swat_test_df)

wadi_train_df = scaler.fit_transform(train_wadi)
wadi_train_df = pd.DataFrame(wadi_train_df)

wadi_test_df = scaler.fit_transform(test_wadi)
wadi_test_df = pd.DataFrame(wadi_test_df)

new_source = []
new_target = []




def kl_(df_A, df_B, kl_list):
    len_A = len(df_A)
    len_B = len(df_B)
    
    for column_A in df_A.columns:
        min_ = min(df_A[column_A])
        max_ = max(df_A[column_A])
        for column_B in df_B.columns:
            min_ = min(df_B[column_B]) if min(df_B[column_B]) < min_ else min_
            max_ = max(df_B[column_B]) if max(df_B[column_B]) > max_ else max_
            
            hist_A = np.histogram(df_A, bins=np.arange(min_, max_, (max_ - min_) * 1))[0]
            hist_B = np.histogram(df_B, bins=np.arange(min_, max_, (max_ - min_) * 1))[0]

            kl_list[column_A][column_B] = float(sum(kl_div(hist_A, hist_B))) #/ len(hist_A)
            
            # if needed
            #plt.hist(hist_A, bins='auto', color='b', label='hist_A')
            #plt.hist(hist_B, bins='auto', color='g', label='hist_B')
            #plt.savefig('./hist_.png')
    kl_list = np.nan_to_num(kl_list, posinf=99999., copy=True)

    return kl_list
            
def feature_importance(X, y):
    xgb = XGB.XGBClassifier(booster='gbtree', importance_type='gain')
    xgb.fit (X, y)
    return xgb.feature_importances_

def leaset_score_based_mapping(kl_list, size, new_source, new_target):
    list_ = np.array(kl_list)

    while len(new_target) < size:
        min_value = min(map(min, list_))
        min_list = np.where(list_ == min_value)

        if len(min_list[0]) > 1:
            idx = random.randrange(len(min_list[0]))
            new_source.append(min_list[0][idx])
            new_target.append(min_list[1][idx])
            print(list_[min_list[0][idx]])
            list_[min_list[0][idx],:] = math.inf
            list_[:,min_list[1][idx]] = math.inf
        else:
            new_source.append(min_list[0][0])
            new_target.append(min_list[1][0])

            list_[min_list[0][0],:] = math.inf
            list_[:,min_list[1][0]] = math.inf
    
def random_mapping(size, new_source, new_target):
    # get array [0, 1, ... , len(size)]
    idx = np.arange(size)
    np.random.shuffle(idx)

    # set new source and target
    new_source = idx.tolist()#[i for i in range(size)]
    np.random.shuffle(idx)
    new_target = idx.tolist()
    
    return new_source, new_target
        
    
    
    
def importance_score_based_mapping(kl_list, col_importance, row_importance, size, new_source, new_target):
    list_ = np.array(kl_list)
    
    # get col importance list
    col_importance_ = col_importance.copy()
    
    while len(new_target) < size:
        # set col max and index of col max   
        col_max = max(col_importance_)
        col_index = col_importance_.index(col_max)
        
        # set row importance list
        row_importance_ = row_importance.copy()
        for idx in new_source:
            row_importance_[idx] = 0

        
        # get minimum value of kl list
        min_value = min(map(min, list_))
        
        for j in range(size):
            # get row max and index of max value
            row_max = max(row_importance_)
            row_index = row_importance_.index(row_max)

            # get min value of list_
            if list_[row_index][col_index] == min_value:
                # add indices to new_source & new_target

                new_source.append(row_index)
                new_target.append(col_index)
                
                # set list_'s indices to inf
                list_[row_index,:] = math.inf
                list_[:,col_index] = math.inf
 
                # set new col importance list
                col_importance_ = col_importance.copy()
                for idx in new_target:
                    col_importance_[idx] = 0
               
                break
            else:
                if row_importance_ == [0] * size:
                    col_importance_[col_index] = 0
                    break
                else:
                    row_importance_[row_index] = 0

        
def feature_reduction(train_A, valid_A, test_A, train_B, valid_B, test_B, label_A, label_B, name_A, name_B, num, method): 
    # method 0 : least score
    # method 1 : random
    # method 2 : feature importance
    new_source = []
    new_target = []

    for i in range(num[0], num[1]):
        columns = [i for i in range(i)]
        
        pca = PCA(n_components=i)
        pca2 = PCA(n_components=i)

        if valid_A is None:
            split_idx = int(len(train_A)/2)
            
            pca_train_A = pca.fit_transform(train_A)
            pca_test_A = pca.transform(test_A)
            pca_valid_A = pca_train_A[-split_idx:]
            sum_a = sum(pca.explained_variance_ratio_)
            print(f'{name_A} pca ratio: {sum_a}')

        elif valid_A:
            raise notImplementedError

        if valid_B is None:
            split_idx = int(len(train_B)/2)
            
            pca_train_B = pca2.fit_transform(train_B)
            pca_test_B = pca2.transform(test_B)
            pca_valid_B = pca_train_B[-split_idx:]
            sum_b = sum(pca2.explained_variance_ratio_)
            print(f'{name_B} pca ratio: {sum_b}')

        elif valid_B:
            raise notImplementedError
            
        else:
            pass
        
        df_valid_A = pd.DataFrame(pca_valid_A, columns=columns)
        df_valid_B = pd.DataFrame(pca_valid_B, columns=columns)
        print("method", method)
        if method == 0:
            # kl divergence
            kl_list = [[0 for col in range(i)] for row in range(i)]
            kl_list = kl_(df_valid_A, df_valid_B, kl_list)
            # mapping

            leaset_score_based_mapping(kl_list, i, new_source, new_target)
        
        elif method == 1:
            # mapping
            new_source, new_target = random_mapping(i, new_source, new_target)

        elif method == 2:
            # kl divergence
            kl_list = [[0 for col in range(i)] for row in range(i)]
            kl_list = kl_(df_valid_A, df_valid_B, kl_list)
        
            # feature importances
            row_importance = feature_importance(pca_test_A, label_A)
            col_importance = feature_importance(pca_test_B, label_B)

            # mapping
            importance_score_based_mapping(kl_list, col_importance.tolist(), row_importance.tolist(), i, new_source, new_target)

        else:
            raise NotImplementedError
        
        print("final source and target")
        print(new_source, new_target)

        # data saving
        train_A = pd.DataFrame(data=pca_train_A, columns=new_source, dtype=float)
        train_B = pd.DataFrame(data=pca_train_B, columns=new_target, dtype=float)
        
        test_A = pd.DataFrame(data=pca_test_A, columns=new_source, dtype=float)
        test_B = pd.DataFrame(data=pca_test_B, columns=new_target, dtype=float)
        
        train_A.to_csv(f'./{name_A}-{name_B}_pca_{name_A}_train_{i}_{method}.csv', sep=',', na_rep='NaN', index=False)
        train_B.to_csv(f'./{name_A}-{name_B}_pca_{name_B}_train_{i}_{method}.csv', sep=',', na_rep='NaN', index=False)
        
        test_A.to_csv(f'./{name_A}-{name_B}_pca_{name_A}_test_{i}_{method}.csv', sep=',', na_rep='NaN', index=False)
        test_B.to_csv(f'./{name_A}-{name_B}_pca_{name_B}_test_{i}_{method}.csv', sep=',', na_rep='NaN', index=False)    