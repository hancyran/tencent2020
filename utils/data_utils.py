import json
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from utils.outlier_utils import Outliers

with open('default_config.json', 'r') as fh:
    cfg = json.load(fh)


def preprocess(is_train=True, is_split=False, log_path=None):
    """
    Preprocess to obtain training dataset

    """
    if is_train:
        if not log_path:
            print('Read Raw Data')
            users, ads, log = read_train_raw_data()

            print('Combine User Log')
            user_log = combine_log(users, ads, log, is_train=is_train, sort_type='time', save_path=log_path)

            print('Remove Outliers')
            user_log = remove_outlier_id(user_log)
        else:
            user_log = pd.read_pickle(os.path.join(cfg['data_path'], log_path))

        if is_split:
            print('Split Dataset')
            train_log, val_log = split_dataset(user_log)

            return train_log, val_log
        else:
            return user_log

    else:
        if not log_path:
            print('Read Raw Data')
            ads, log = read_test_raw_data()

            print('Combine User Log')
            user_log = combine_log(ads, log, is_train=is_train, sort_type='time', save_path=log_path)

            print('Remove Outliers')
            user_log = remove_outlier_id(user_log, is_train=is_train)
        else:
            user_log = pd.read_pickle(os.path.join(cfg['data_path'], log_path))

        return user_log


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


def combine_log(ads, log, users=None, is_train=True, save_path=None, sort_type='time'):
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

    def combine_log(merged_log):
        # combine id into one sequence
        def combine_id(x):
            col = list(set(x))
            col.sort(key=list(x).index)
            return ','.join([str(i) for i in col])

        # creative_id
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

        return user_log

    if sort_type == 'time':
        # sort by time
        merged_log.sort_values(by='time', inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        user_log = combine_log(merged_log)

    elif sort_type == 'click_times':
        # count click times
        group = merged_log[['user_id', 'creative_id', 'click_times']].groupby(['user_id', 'creative_id']).sum()
        group.reset_index()

        # merge count times
        merged_log.drop(['click_times'], inplace=True)
        merged_log = pd.merge(merged_log, group, on=['user_id', 'creative_id'])

        # sort by click times
        merged_log.sort_values(by='click_times', ascending=False, inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        user_log = combine_log(merged_log)

    elif sort_type == 'both_combined':
        # sort by time
        merged_log.sort_values(by='time', ascending=True, inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        time_log = combine_log(merged_log)

        # count click times
        group = merged_log[['user_id', 'creative_id', 'click_times']].groupby(['user_id', 'creative_id']).sum()
        group.reset_index()

        # merge count times
        merged_log.drop(['click_times'], inplace=True)
        merged_log = pd.merge(merged_log, group, on=['user_id', 'creative_id'])

        # sort by click times
        merged_log.sort_values(by='click_times', ascending=False, inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        click_log = combine_log(merged_log)

        # merge both types
        user_log = pd.merge(time_log, click_log, on='user_id', suffixes=('_time', '_click'))

    elif sort_type == 'both_sorted':
        # sort by time(ascending) & click_times(descending)
        merged_log.sort_values(by=['time', 'click_times'], ascending=[True, False], inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        user_log = combine_log(merged_log)

    # merge labels
    if is_train:
        user_log = pd.merge(user_log, users, on=['user_id', 'age', 'gender'])

    if save_path:
        user_log.to_pickle(os.path.join(cfg['data_path'], save_path))

    return user_log


def remove_outlier_id(log, is_train=True, save_path=None):
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

    if save_path:
        user_log.to_pickle(os.path.join(cfg['data_path'], save_path))

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
    train_log, val_log = split_dataset(user_log)

    print(train_log.shape)
    print(val_log.shape)
