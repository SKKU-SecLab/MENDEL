import pandas as pd
import numpy as np
import sys
import pickle as pkl
import math
from pathlib import Path

def preprocessing(data, prefix):
	if data == 'SWaT':
		# read original datasets
		train = pd.read_csv(os.path.join(prefix, 'SWaT_Dataset_Normal_v1_0.csv'))
		test = pd.read_csv(os.path.join(prefix, 'SWaT_Dataset_test.csv'))

		# split into train&test&label
		train = train.drop([' Timestamp', 'label'], axis=1)
		label = test['label']
		test = test.drop([' Timestamp', 'label'], axis=1)

		# translate csv to pickle
		with open('SWaT_train.pkl', 'wb') as f:
			pkl.dump(train, f)
		with open('SWaT_test.pkl', 'wb') as f:
			pkl.dump(test, f)
		with open('SWaT_label.pkl', 'wb') as f:
			pkl.dump(label, f)

		# pickle test
		with open('SWaT_train.pkl', 'rb') as f:
			train_ = pkl.load(f)
		with open('SWaT_test.pkl', 'rb') as f:
			test_ = pkl.load(f)
		with open('SWaT_label.pkl', 'rb') as f:
			label_ = pkl.load(f)
		print("train shape : ", train_.shape)
		print("test shape : ", test_.shape)
		print("label shape : ", label_.shape)
		print("SWaT finished")


	elif data == 'WADI':
		return
	else:
		raise NotImplementedError("Only support SWaT and WADI")
	return




### SWaT
src = "./datasets/Pickles/SWaT/"

### WADI
src = "./datasets/Pickles/WADI/"



def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])


#TRAIN_DATASET = sorted([x for x in Path("./hai21").glob("*.csv")])
#TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)


#train_swat_a = pd.read_csv("./swat_a/SWaT_Dataset_Normal_v1_0.csv")
#test_swat_a = pd.read_csv("./swat_a/SWaT_Dataset_test.csv")
#label_swat_a = test_swat_a['label']
#train_swat_a = train_swat_a.drop([' Timestamp', 'label'], axis=1)
#test_swat_a = test_swat_a.drop([' Timestamp', 'label'], axis=1)

#test_swat_a.to_csv('./test.csv', header=True, index=False)
#train_swat_a.to_csv('/data/jeonyong/DeepADoTS/data/raw/SWaT/SWaT_train.csv', header=True, index=False)
#test_swat_a.to_csv('/data/jeonyong/DeepADoTs/data/raw/SWaT/SWaT_test.csv', header=False, index=False)

'''



### WADI
src_path = "./wadi/"

# WADI A2
train_wadi = pd.read_csv(f'{src_path}A2/WADI_train_remove_NaN.csv')
test_wadi = pd.read_csv(f'{src_path}A2/WADI_attack_remove_NaN.csv')

test_wadi.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)':'label'}, inplace=True)
test_wadi['label'].replace({1:0}, inplace=True)
test_wadi['label'].replace({-1:1}, inplace=True)

train_wadi.dropna(axis=0, inplace=True)
test_wadi.dropna(axis=0, inplace=True)

train_wadi.drop(['Date', 'Time'], axis=1, inplace=True)
test_wadi.drop(['Date', 'Time'], axis=1, inplace=True)
print(test_wadi.shape)

with open('WADI_A2_train.pkl', 'wb') as f:
	pkl.dump(train_wadi, f)

with open('WADI_A2_test.pkl', 'wb') as f:
	pkl.dump(test_wadi, f)

# WADI A2
#print(train_wadi)
#print(TRAIN_DF_RAW['timestamp'][:784537])

#new_df = pd.concat([TRAIN_DF_RAW['timestamp'][:784537], train_wadi], ignore_index=True)
#TRAIN_DF_RAW['timestamp'][:172801].to_csv("./test.csv")
#print(new_df)

# WADI A1
src_path = "./wadi/"
train_wadi = pd.read_csv(f'{src_path}A1/WADI_train_remove_NaN.csv')
test_wadi = pd.read_csv(f'{src_path}A1/WADI_attack_remove_NaN.csv')


#train_wadi.dropna(axis=1, inplace=True)
#test_wadi.dropna(axis=1, inplace=True)
train_wadi.dropna(axis=0, inplace=True)
test_wadi.dropna(axis=0, inplace=True)

label_wadi = test_wadi['label']
#A1_columns = list(set(test_wadi.columns) - set(train_wadi.columns) - set(['label']))
#A1_columns.append('Date')
#A1_columns.append('Time')
#A1_columns.append('label')

train_wadi.drop(['Date', 'Time'], axis=1, inplace=True)
test_wadi.drop(['Date', 'Time', 'label'], axis=1, inplace=True)

print(train_wadi.shape)
print(test_wadi.shape)
print(label_wadi.shape)
with open('./wadi/A1/WADI_A1_train.pkl', 'wb') as f:
	pkl.dump(train_wadi, f)
	f.close()

with open('./wadi/A1/WADI_A1_test.pkl', 'wb') as f:
	pkl.dump(test_wadi, f)
	f.close()

with open('./wadi/A1/WADI_A1_test_label.pkl', 'wb') as f:
	pkl.dump(label_wadi, f)
	f.close()
# WADI A1


train_wadi = pd.read_csv('./swat_wadi/pca_wadi_A1_train_30.csv')
test_wadi = pd.read_csv('./swat_wadi/pca_wadi_A1_test_30.csv')
#label_wadi = pd.read_csv('./wadi/A2/WADI_attack_remove_NaN.csv')
#label_wadi = label_wadi['label']
print(train_wadi.shape, test_wadi.shape)
#print(label_wadi.shape)

#with open("../InterFusion/data/processed/WADI_test_label.pkl", "wb") as f:
#	pkl.dump(label_wadi, f)
#	f.close()

with open('WADI_A1_train_30.pkl', 'wb') as f:
	pkl.dump(train_wadi, f)
	f.close()

with open('WADI_A1_test_30.pkl', 'wb') as f:
	pkl.dump(test_wadi, f)
	f.close()

train_wadi = pd.read_csv('./swat_wadi/pca_wadi_train_20.csv')
test_wadi = pd.read_csv('./swat_wadi/pca_wadi_test_20.csv')

with open('WADI_train_20.pkl', 'wb') as f:
	pkl.dump(train_wadi, f)
	f.close()

with open('WADI_test_20.pkl', 'wb') as f:
	pkl.dump(test_wadi, f)
	f.close()

train_swat = pd.read_csv('./swat_wadi/pca_wadi_A1_train_20.csv')
test_swat = pd.read_csv('./swat_wadi/pca_wadi_A1_test_20.csv')

with open('WADI_A1_train_20.pkl', 'wb') as f:
	pkl.dump(train_swat, f)
	f.close()
with open('WADI_A1_test_20.pkl', 'wb') as f:
	pkl.dump(test_swat, f)
	f.close()

train_swat = pd.read_csv('./swat_wadi/pca_swat_A1_train_20.csv')
test_swat = pd.read_csv('./swat_wadi/pca_swat_A1_test_20.csv')

with open('SWaT_A1_train_20.pkl', 'wb') as f:
	pkl.dump(train_swat, f)
	f.close()
with open('SWaT_A1_test_20.pkl', 'wb') as f:
	pkl.dump(test_swat, f)
	f.close()


with open('SWaT_A1_train_20.pkl', 'rb') as f:
	data = pkl.load(f)
	print(data.shape)

with open('WADI_A1_test_20.pkl', 'rb') as f:
	data = pkl.load(f)
	print(data.shape)
"""

#train = pd.read_csv('mapping_data/algo/pca_swat_train_20.csv')
#test = pd.read_csv('mapping_data/algo/pca_swat_test_20.csv')

#with open('SWaT_least_train_20.pkl', 'wb') as f:
#	pkl.dump(train, f)

#with open('SWaT_least_test_20.pkl', 'wb') as f:
#	pkl.dump(test, f)


"""

swat_train = pd.read_csv('./mapping_data/valid_based/feature_importance/1/pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/feature_importance/1/pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/feature_importance/1/pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/feature_importance/1/pca_wadi_test_20.csv')

with open('SWaT_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open('SWaT_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open('WADI_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open('WADI_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)


swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/1/pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/1/pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/1/pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/1/pca_wadi_test_20.csv')

with open('SWaT_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open('SWaT_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open('WADI_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open('WADI_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)


swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/1/pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/1/pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/1/pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/1/pca_wadi_test_20.csv')

with open('SWaT_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open('SWaT_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open('WADI_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open('WADI_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/1/pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/1/pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/1/pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/1/pca_wadi_test_20.csv')

with open('SWaT_random_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open('SWaT_random_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open('WADI_random_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open('WADI_random_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)


swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/1/pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/1/pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/1/pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/1/pca_wadi_test_20.csv')

with open('SWaT_random_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open('SWaT_random_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open('WADI_random_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open('WADI_random_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/1/pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/1/pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/1/pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/1/pca_wadi_test_20.csv')

with open('SWaT_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open('SWaT_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open('WADI_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open('WADI_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/1/pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/1/pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/1/pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/1/pca_wadi_test_20.csv')

with open('SWaT_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open('SWaT_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open('WADI_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open('WADI_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)
"""

# USAD
### WADI-SWaT
swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

### SWaT-WADI
swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/8(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/8(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/8(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/8(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

# InterFusion
### WADI-SWaT
swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

### SWaT-WADI
swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/8(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/8(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/8(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/8(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/6(25 features)/pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/6(25 features)/pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/6(25 features)/pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/6(25 features)/pca_wadi_test_25.csv')

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)
'''

### wadi-swat
'''
swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_swat_train_10.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_swat_test_10.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_wadi_train_10.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_wadi_test_10.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)


swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_swat_train_10.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_swat_test_10.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_wadi_train_10.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_wadi_test_10.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_train_10.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_test_10.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_train_10.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_test_10.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_swat_train_15.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_swat_test_15.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_wadi_train_15.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_wadi_test_15.csv')

### 15
with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_swat_train_15.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_swat_test_15.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_wadi_train_15.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_wadi_test_15.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_train_15.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_test_15.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_train_15.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_test_15.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

### 20

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_wadi_test_20.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_wadi_test_20.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_test_20.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

### 25

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/random/wadi-swat_pca_wadi_test_25.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_random_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_random_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_random_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_random_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/least_score/wadi-swat_pca_wadi_test_25.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_test_25.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

### swat-wadi

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_swat_train_10.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_swat_test_10.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_wadi_train_10.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_wadi_test_10.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_swat_train_10.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_swat_test_10.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_wadi_train_10.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_wadi_test_10.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_train_10.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_test_10.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_train_10.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_test_10.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_train_10.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_test_10.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_train_10.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_test_10.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)


swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_swat_train_15.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_swat_test_15.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_wadi_train_15.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_wadi_test_15.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_swat_train_15.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_swat_test_15.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_wadi_train_15.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_wadi_test_15.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_train_15.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_test_15.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_train_15.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_test_15.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_train_15.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_test_15.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_train_15.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_test_15.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_wadi_test_20.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_wadi_test_20.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_train_20.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_test_20.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_train_20.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_test_20.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_train_20.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_test_20.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/random/swat-wadi_pca_wadi_test_25.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_random_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_random_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_random_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_random_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/least_score/swat-wadi_pca_wadi_test_25.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_least_score_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_train_25.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_test_25.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_train_25.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_test_25.csv')

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_train_25.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_test_25.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)
'''
#####################################################################################################################################bin 0.1


swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_train_30.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_test_30.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_train_30.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_test_30.csv')

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_train_30.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_test_30.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_train_30.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_test_30.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-WADI_SWaT_train_30.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-WADI_SWaT_test_30.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-WADI_WADI_train_30.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-WADI_WADI_test_30.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

#with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(swat_train, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(swat_test, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#
#
swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_train_30.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_test_30.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_train_30.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_test_30.csv')

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_train_30.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_test_30.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_train_30.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_test_30.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data/jeonyong/InterFusion/data/processed/WADI-SWaT_SWaT_train_30.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/processed/WADI-SWaT_SWaT_test_30.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/processed/WADI-SWaT_WADI_train_30.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/processed/WADI-SWaT_WADI_test_30.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

#with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(swat_train, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(swat_test, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#
#
swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_train_35.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_swat_test_35.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_train_35.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/SWaT-WADI/feature_importance/swat-wadi_pca_wadi_test_35.csv')

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_train_35.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_SWaT_test_35.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_train_35.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/SWaT-WADI/SWaT-WADI_WADI_test_35.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-WADI_SWaT_train_35.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-WADI_SWaT_test_35.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-WADI_WADI_train_35.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-WADI_WADI_test_35.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

#with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(swat_train, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/SWaT-WADI/SWaT_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(swat_test, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/SWaT-WADI/WADI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#
#
swat_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_train_35.csv')
swat_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_swat_test_35.csv')

wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_train_35.csv')
wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-SWaT/feature_importance/wadi-swat_pca_wadi_test_35.csv')

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_train_35.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_SWaT_test_35.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_train_35.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/WADI-SWaT/WADI-SWaT_WADI_test_35.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

with open(f'/data/jeonyong/InterFusion/data/processed/WADI-SWaT_SWaT_train_35.pkl', 'wb') as f:
	pkl.dump(swat_train, f)

with open(f'/data/jeonyong/InterFusion/data/processed/WADI-SWaT_SWaT_test_35.pkl', 'wb') as f:
	pkl.dump(swat_test, f)

with open(f'/data/jeonyong/InterFusion/data/processed/WADI-SWaT_WADI_train_35.pkl', 'wb') as f:
	pkl.dump(wadi_train, f)

with open(f'/data/jeonyong/InterFusion/data/processed/WADI-SWaT_WADI_test_35.pkl', 'wb') as f:
	pkl.dump(wadi_test, f)

#with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(swat_train, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/SWaT/WADI-SWaT/SWaT_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(swat_test, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#
#with open(f'/data_ssd/jeongyong/inter_usad/input/WADI/WADI-SWaT/WADI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

### HAI-WADI/SWaT test

#wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_train_10_2.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_test_10_2.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_10_2.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_10_2.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/WADI-HAI_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/WADI-HAI_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/WADI-HAI_WADI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/WADI-HAI_WADI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)


#wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_train_15.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_test_15.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_15.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_15.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

#wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_train_20.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_test_20.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_20.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_20.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

#wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_train_25.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_test_25.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_25.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_25.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_test_25,pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

#wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_train_30.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_test_30.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_30.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_30.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

#wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_train_35.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_test_35.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_35.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_35.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_HAI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-WADI/HAI-WADI_WADI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#
#swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_train_10_2.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_test_10_2.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_train_10_2.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_test_10_2.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-HAI_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-HAI_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)	
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-HAI_SWaT_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-HAI_SWaT_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_train_15.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_test_15.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_train_15.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_test_15.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

#swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_train_20.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_test_20.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_train_20.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_test_20.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

#swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_train_25.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_test_25.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_train_25.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_test_25.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

#swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_train_30.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_test_30.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_train_30.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_test_30.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

#swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_train_35.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_wadi_test_35.csv')
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_train_35.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_wadi_test_35.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_HAI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(wadi_train, f)
#with open(f'/data/jeonyong/InterFusion/data/HAI-SWaT/HAI-SWaT_SWaT_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(wadi_test, f)

### HAI 21

#hai_train = sorted([x for x in Path("./mapping_data/hai21/train/").glob("*.csv")])
#hai_train = dataframe_from_csvs(hai_train)
#
#hai_test = sorted([x for x in Path("./mapping_data/hai21/test/").glob("*.csv")])
#hai_test = dataframe_from_csvs(hai_test)
#
#hai_train = hai_train.drop(["time", "attack", "attack_P1", "attack_P2", "attack_P3"], axis=1)
#hai_test = hai_test.drop(["time", "attack_P1", "attack_P2", "attack_P3"], axis=1)
#
#hai_label = hai_test["attack"]
#hai_test = hai_test.drop(["attack"], axis=1)
#
#print(hai_test)
#print(hai_label)

#with open(f'/data/jeonyong/InterFusion/data/processed/HAI_train.pkl', 'wb') as f:
#	pkl.dump(hai_train, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI_test.pkl', 'wb') as f:
#	pkl.dump(hai_test, f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI_test_label.pkl', 'wb') as f:
#	pkl.dump(hai_label, f)

###

#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_train_10.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_test_10.csv')
#
#swat_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_train_10.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_test_10.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_train_15.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_test_15.csv')
#
#swat_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_train_15.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_test_15.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)	
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_train_20.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_test_20.csv')
#
#swat_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_train_20.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_test_20.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_train_25.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_test_25.csv')
#
#swat_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_train_25.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_test_25.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_train_30.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_test_30.csv')
#
#swat_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_train_30.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_test_30.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_test310.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_train_35.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_hai_test_35.csv')
#
#swat_train = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_train_35.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/HAI-SWaT/swat-hai_pca_swat_test_35.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_HAI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-SWaT_SWaT_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-SWaT/HAI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/inter_usad/input/SWaT/HAI-SWaT/SWaT_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
####
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_train_10.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_test_10.csv')
#
#wadi_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_10.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_10.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_train_10.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_test_10.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_train_15.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_test_15.csv')
#
#wadi_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_15.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_15.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_train_20.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_test_20.csv')
#
#wadi_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_20.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_20.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_train_20.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_train_25.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_test_25.csv')
#
#wadi_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_25.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_25.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_train_30.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_test_30.csv')
#
#wadi_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_30.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_30.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#hai_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_train_35.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_hai_test_35.csv')
#
#wadi_train = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_train_35.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/HAI-WADI/wadi-hai_pca_wadi_test_35.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_HAI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/InterFusion/data/processed/HAI-WADI_WADI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/inter_usad/input/HAI/HAI-WADI/HAI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/inter_usad/input/WADI/HAI-WADI/WADI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#
##hai_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_train_10.csv')
##hai_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_test_10.csv')
##
##swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_train_10.csv')
##swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_test_10.csv')
#
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_10.pkl', 'wb') as f:
##	pkl.dump(hai_train,f)
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_10.pkl', 'wb') as f:
##	pkl.dump(hai_test,f)
##with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-HAI_HAI_feature_importance_train_10.pkl', 'wb') as f:
##	pkl.dump(hai_train,f)
##with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-HAI_HAI_feature_importance_test_10.pkl', 'wb') as f:
##	pkl.dump(hai_test,f)
#
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_10.pkl', 'wb') as f:
##	pkl.dump(swat_train,f)
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_10.pkl', 'wb') as f:
##	pkl.dump(swat_test,f)
##with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-HAI_SWaT_feature_importance_train_10.pkl', 'wb') as f:
##	pkl.dump(swat_train,f)
##with open(f'/data/jeonyong/InterFusion/data/processed/SWaT-HAI_SWaT_feature_importance_test_10.pkl', 'wb') as f:
##	pkl.dump(swat_test,f)
#
##hai_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_train_15.csv')
##hai_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_test_15.csv')
##
##swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_train_15.csv')
##swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_test_15.csv')
##
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_15.pkl', 'wb') as f:
##	pkl.dump(hai_train,f)
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_15.pkl', 'wb') as f:
##	pkl.dump(hai_test,f)
##
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_15.pkl', 'wb') as f:
##	pkl.dump(swat_train,f)
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_15.pkl', 'wb') as f:
##	pkl.dump(swat_test,f)
##
##
##hai_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_train_20.csv')
##hai_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_test_20.csv')
##
##swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_train_20.csv')
##swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_test_20.csv')
##
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_20.pkl', 'wb') as f:
##	pkl.dump(hai_train,f)
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_20.pkl', 'wb') as f:
##	pkl.dump(hai_test,f)
##
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_20.pkl', 'wb') as f:
##	pkl.dump(swat_train,f)
##with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_20.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#
#hai_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_train_25.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_test_25.csv')
#
#swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_train_25.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_test_25.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_25.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_25.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#
#hai_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_train_30.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_test_30.csv')
#
#swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_train_30.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_test_30.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_30.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_30.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#
#hai_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_train_35.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_hai_test_35.csv')
#
#swat_train = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_train_35.csv')
#swat_test = pd.read_csv('./mapping_data/valid_based/SWaT-HAI/swat-hai_pca_swat_test_35.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_HAI_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_train_35.pkl', 'wb') as f:
#	pkl.dump(swat_train,f)
#with open(f'/data/jeonyong/InterFusion/data/SWaT-HAI/SWaT-HAI_SWaT_feature_importance_test_35.pkl', 'wb') as f:
#	pkl.dump(swat_test,f)
#
#
####
#
#hai_train = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_hai_train_15.csv')
#hai_test = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_hai_test_15.csv')
#
#wadi_train = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_train_15.csv')
#wadi_test = pd.read_csv('./mapping_data/valid_based/WADI-HAI/wadi-hai_pca_wadi_test_15.csv')
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(hai_train,f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_HAI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(hai_test,f)
#
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_train_15.pkl', 'wb') as f:
#	pkl.dump(wadi_train,f)
#with open(f'/data/jeonyong/InterFusion/data/WADI-HAI/WADI-HAI_WADI_feature_importance_test_15.pkl', 'wb') as f:
#	pkl.dump(wadi_test,f)
#