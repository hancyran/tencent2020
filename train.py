import json
import numpy as np
import pandas as pd
from utils import model_utils
import tensorflow as tf
import utils.misc_utils as utils
import os
import gc
from sklearn import metrics
from sklearn import preprocessing
import random

from utils.feature_utils import Features

with open('default_config.json', 'r') as fh:
    cfg = json.load(fh)
np.random.seed(2019)

####################################################################################
feats = Features()

# hyper params
hparam = tf.contrib.training.HParams(
    model='CIN',
    norm=True,
    batch_norm_decay=0.9,
    hidden_size=[1024, 512],
    dense_hidden_size=[300],
    cross_layer_sizes=[128, 128],
    k=16,
    single_k=16,
    max_length=100,
    cross_hash_num=int(5e6),
    single_hash_num=int(5e6),
    multi_hash_num=int(1e6),
    batch_size=32,
    infer_batch_size=2 ** 14,
    optimizer="adam",
    dropout=0,
    kv_batch_num=20,
    learning_rate=0.0002,
    num_display_steps=1000,
    num_eval_steps=1000,
    epoch=1,  # don't modify
    metric='SMAPE',
    activation=['relu', 'relu', 'relu'],
    init_method='tnormal',
    cross_activation='relu',
    init_value=0.001,
    single_features=feats.single_features,
    cross_features=feats.cross_features,
    multi_features=feats.multi_features,
    dense_features=feats.dense_features,
    kv_features=feats.kv_features,
    label='imp',
    model_name="CIN")
utils.print_hparams(hparam)

####################################################################################

# read data
test = pd.read_pickle('data/test_NN.pkl')
dev = pd.read_pickle('data/dev_NN.pkl')
train = pd.read_pickle('data/train_NN_0.pkl')
train_dev = pd.read_pickle('data/train_dev_NN_0.pkl')
train['gold_imp'] = train['imp']
dev['gold_imp'] = dev['imp']
train['imp'] = train['imp'].apply(lambda x: np.log(x + 1))
train_dev['imp'] = train_dev['imp'].apply(lambda x: np.log(x + 1))

####################################################################################

# train & validate
print(dev.shape)
print(train_dev.shape)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 8))
scaler.fit(train_dev[['imp']])
hparam.train_scaler = scaler
hparam.test_scaler = scaler
print("*" * 80)
model = model_utils.build_model(hparam)
model.train(train_dev, dev)
dev_preds = np.zeros(len(dev))
dev_preds = model.infer(dev)
dev_preds = np.exp(dev_preds) - 1
print(np.mean(dev_preds))
print("*" * 80)

####################################################################################

# test
print(test.shape)
print(train.shape)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 8))
scaler.fit(train[['imp']])
hparam.train_scaler = scaler
hparam.test_scaler = scaler
index = set(range(train.shape[0]))
K_fold = []
for i in range(5):
    if i == 4:
        tmp = index
    else:
        tmp = random.sample(index, int(1.0 / 5 * train.shape[0]))
    index = index - set(tmp)
    print("Number:", len(tmp))
    K_fold.append(tmp)

train_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []
train['gold'] = True
for i in range(5):
    print("Fold", i)
    dev_index = K_fold[i]
    train_index = []
    for j in range(5):
        if j != i:
            train_index += K_fold[j]
    for k in range(2):
        model = model_utils.build_model(hparam)
        score = model.train(train.loc[train_index], train.loc[dev_index])
        scores.append(score)
        train_preds[list(dev_index)] += model.infer(train.loc[list(dev_index)]) / 2
        test_preds += model.infer(test) / 10
        print(np.mean((np.exp(test_preds * 10 / (i * 2 + k + 1)) - 1)))
    try:
        del model
        gc.collect()
    except:
        pass
train_preds = np.exp(train_preds) - 1
test_preds = np.exp(test_preds) - 1

####################################################################################

# output
print(scores)
print(np.mean(scores))
print(train_preds.mean())
print(dev_preds.mean())
print(test_preds.mean())
train_fea = train[['aid', 'request_day']]
train_fea['nn_preds'] = train_preds
dev['nn_preds'] = dev_preds
dev_fea = dev[['aid', 'bid', 'gold', 'imp', 'nn_preds']]
test['nn_preds'] = test_preds
test_fea = test[['aid', 'nn_preds']]
train_fea.to_csv('submission/nn_pred_{}_train.csv'.format(hparam.model_name), index=False)
test_fea.to_csv('submission/nn_pred_{}_test.csv'.format(hparam.model_name), index=False)
dev_fea.to_csv('submission/nn_pred_{}_dev.csv'.format(hparam.model_name), index=False)
####################################################################################
