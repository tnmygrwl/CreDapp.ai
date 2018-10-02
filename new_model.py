import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.models import load_model
import h5py


# import os
# print(os.listdir("../input"))
# print(os.listdir("../input/save-dromosys-features"))
'''
df = pd.read_pickle('../df.pkl.gz')
print("Raw shape: ", df.shape)

y = df['TARGET']
feats = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
X = df[feats]
print("X shape: ", X.shape, "    y shape:", y.shape)

print("\nPreparing data...")
X = X.fillna(X.mean()).clip(-1e11, 1e11)
scaler = MinMaxScaler()
scaler.fit(X)
# training = y.notnull()
testing = y.isnull()
# X_train = scaler.transform(X[training])
'''
scaler = MinMaxScaler()
X_testing_raw = pd.read_csv("~/Downloads/X_testing.csv")
X_testing = X_testing_raw.fillna(X_testing_raw.mean()).clip(-1e11, 1e11)
scaler.fit(X_testing)
print(X_testing.shape)
X_test = scaler.transform(X_testing)
# y_train = np.array(y[training])
# print(X_train.shape, X_test.shape, y_train.shape)

# print('Setting up neural network...')
'''
nn = Sequential()
nn.add(Dense(units=400, kernel_initializer='normal', input_dim=718))
nn.add(PReLU())
nn.add(Dropout(.3))
nn.add(Dense(units=160, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(units=64, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(units=26, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(units=12, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
nn.compile(loss='binary_crossentropy', optimizer='adam')
print('Fitting neural network...')
# nn.fit(X_train, y_train, validation_split=0.1, epochs=10, verbose=2)
'''
nn = load_model('my_model.h5')
# nn.compile(loss='binary_crossentropy', optimizer='adam')
X_test.flatten().clip(0, 1)
print('Predicting...')
y_pred = nn.predict(X_test).flatten().clip(0, 1)

print('Saving results...')
sub = pd.DataFrame()
sub['SK_ID_CURR'] = X_testing_raw['SK_ID_CURR']
sub['TARGET'] = y_pred
sub[['SK_ID_CURR', 'TARGET']].to_csv('sub_nn.csv', index=False)

print(sub.head())
