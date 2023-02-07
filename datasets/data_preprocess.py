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
		train_wadi = pd.read_csv(os.path.join(prefix, 'WADI_14days.csv'))
		test_wadi = pd.read_csv(os.path.join(prefix, 'WADI_attackdata.csv'))
		
		train_wadi.dropna(axis=1, inplace=True)
		test_wadi.dropna(axis=1, inplace=True)
		train_wadi.dropna(axis=0, inplace=True)
		test_wadi.dropna(axis=0, inplace=True)
		
		label_wadi = test_wadi['label']

		train_wadi.drop(['Date', 'Time'], axis=1, inplace=True)
		test_wadi.drop(['Date', 'Time', 'label'], axis=1, inplace=True)
		
		with open('WADI_train.pkl', 'wb') as f:
			pkl.dump(train_wadi, f)
		
		with open('WADI_test.pkl', 'wb') as f:
			pkl.dump(test_wadi, f)
		
		with open('WADI_label.pkl', 'wb') as f:
			pkl.dump(label_wadi, f)

		with open('WADI_train.pkl', 'rb') as f:
			train_ = pkl.load(f)
		with open('WADI_test.pkl', 'rb') as f:
			test_ = pkl.load(f)
		with open('WADI_label.pkl', 'rb') as f:
			label_ = pkl.load(f)
		print("train shape : ", train_.shape)
		print("test shape : ", test_.shape)
		print("label shape : ", label_.shape)
		print("WADI finished")

	else:
		raise NotImplementedError("Only support SWaT and WADI")
	return

