import json
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from utils.outlier_utils import Outliers

with open('default_config.json', 'r') as fh:
    cfg = json.load(fh)


def read_train_raw_data(root_path='data/train'):
    """
    Read Train Raw Data

    """
    users = pd.read_csv(os.path.join(root_path, 'user.csv'))
    ads = pd.read_csv(os.path.join(root_path, 'ad.csv'))
    log = pd.read_csv(os.path.join(root_path, 'click_log.csv'))

    # turn to null value
    ads.replace('\\N', 0, inplace=True)
    ads.product_id = ads.product_id.astype(np.int64)
    ads.industry = ads.industry.astype(np.int64)

    return users, ads, log


def read_test_raw_data(root_path='data/test'):
    """
    Read Test Raw Data

    """
    ads = pd.read_csv(os.path.join(root_path, 'ad.csv'))
    log = pd.read_csv(os.path.join(root_path, 'click_log.csv'))

    # turn to null value
    ads.replace('\\N', 0, inplace=True)
    ads.product_id = ads.product_id.astype(np.int64)
    ads.industry = ads.industry.astype(np.int64)

    return ads, log


def combine_log(ads, log, users=None, is_train=True):
    """
    Combine Log into User-Primary Log

    TODO Add Click Times & Time
    """
    # merge df
    if is_train:
        merged_log = pd.merge(log, users, on='user_id')
        merged_log = pd.merge(merged_log, ads, on='creative_id')
    else:
        merged_log = pd.merge(log, ads, on='creative_id')

    # combine id into one entity
    # creative_id
    combine_id = lambda x: ','.join([str(i) for i in list(set(x.values))])
    combined_log = merged_log[['user_id', 'creative_id']].groupby(['user_id']).agg({'creative_id': combine_id})
    user_log = combined_log.reset_index()
    # ad_id
    combined_log = merged_log[['user_id', 'ad_id']].groupby(['user_id']).agg({'ad_id': combine_id})
    user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
    # product_id
    combined_log = merged_log[['user_id', 'product_id']].groupby(['user_id']).agg({'product_id': combine_id})
    user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
    # product_category
    combined_log = merged_log[['user_id', 'product_category']].groupby(['user_id']).agg(
        {'product_category': combine_id})
    user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
    # advertiser_id
    combined_log = merged_log[['user_id', 'advertiser_id']].groupby(['user_id']).agg({'advertiser_id': combine_id})
    user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
    # industry
    combined_log = merged_log[['user_id', 'industry']].groupby(['user_id']).agg({'industry': combine_id})
    user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')

    if is_train:
        user_log = pd.merge(user_log, users, on='user_id')

    return user_log


def remove_outlier_id(log, is_train=True):
    """
    Remove Outliers with existing user_id

    """
    # fetch outlier ids
    outliers = Outliers()
    if is_train:
        outlier_ids = outliers.train_userid_outliers
    else:
        outlier_ids = outliers.test_userid_outliers

    # remove outliers
    for i in outlier_ids:
        log = log.drop(log[log.user_id == i].index)

    return log


def split_dataset(log):
    """
    Split Data into Validation and Training Dataset

    """
    train_log, val_log = train_test_split(log, test_size=0.05, random_state=cfg['seed'], shuffle=True)

    return train_log, val_log


if __name__ == '__main__':
    print('Read Raw Data')
    users, ads, log = read_train_raw_data()

    print('Combine User Log')
    user_log = combine_log(users, ads, log)

    print('Remove Outliers')
    user_log = remove_outlier_id(user_log)

    print('Split Dataset')
    train_log, val_log = remove_outlier_id(user_log)

    print(train_log.shape)
    print(val_log.shape)
