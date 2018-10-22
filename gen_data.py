#!/usr/bin/env python
# encoding: utf-8
from feature_extract import *
import pandas as pd
import numpy as np
def add_distance_rate(df):
    # 距离特征
    # how to deal with distance == null? mean?
    # mean = df[df['Distance'] != 'null']['Distance'].astype(float).mean()
    # print 'distance_rate_mean: ', mean
    frame = df['Distance'].map(lambda x: float(x) / 10 if x != 'null' else -1)
    frame.name = 'distance_rate'
    df = df.join(frame)
    return df
def coupon_type(df):
    # 优惠券类型
    frame = df['Discount_rate'].map(lambda x: -1 if x == 'null' else 0 if str(x).find(':') == -1 else 1)
    frame.name = 'coupon_type'
    return frame


def coupon_discount(df):
    # 优惠券折率
    frame = df['Discount_rate'].map(discount_rate_calculation)
    frame.name = 'coupon_discount'
    return frame


def coupon_discount_floor(df):
    # 优惠券满减下限
    frame = df['Discount_rate'].map(discount_floor_partition)
    frame.name = 'coupon_discount_floor'
    return frame

coupon_features = [coupon_type, coupon_discount, coupon_discount_floor]
def add_coupon_features(df):
    for f in coupon_features:
        df = df.join(f(df))
    return df

def user_received_counts(df):
    # 用户领取的所有优惠券数目及归一化
    frame = df['Coupon_id'].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'user_received_counts'
    received_users = df[['User_id']].join(frame)
    grouped = received_users.groupby('User_id', as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'user_received_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on='User_id', how='left')
    return df


def user_received_coupon_counts(df):
    # 用户领取的特定优惠券数目及归一化
    frame = df['Coupon_id'].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'user_received_coupon_counts'
    received_users = df[['User_id', 'Coupon_id']].join(frame)
    grouped = received_users.groupby(['User_id', 'Coupon_id'], as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'user_received_coupon_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=['User_id', 'Coupon_id'], how='left')
    return df


def merchant_received_counts(df):
    # 商家被领取的优惠券数目及归一化
    frame = df['Coupon_id'].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'merchant_received_counts'
    received_merchants = df[['Merchant_id']].join(frame)
    grouped = received_merchants.groupby('Merchant_id', as_index=False).sum()
    normalized_frame = min_max_normalize(received_merchants, frame.name)[frame.name]
    normalized_frame.name = 'merchant_received_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on='Merchant_id', how='left')
    return df


def merchant_received_coupon_counts(df):
    # 商家被领取的特定优惠券数目及归一化
    frame = df['Coupon_id'].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'merchant_received_coupon_counts'
    received_merchants = df[['Merchant_id', 'Coupon_id']].join(frame)
    grouped = received_merchants.groupby(['Merchant_id', 'Coupon_id'], as_index=False).sum()
    normalized_frame = min_max_normalize(received_merchants, frame.name)[frame.name]
    normalized_frame.name = 'merchant_received_coupon_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=['Merchant_id', 'Coupon_id'], how='left')
    return df


def user_merchant_received_counts(df):
    # 用户领取特定商家的优惠券数目及归一化
    frame = df['Coupon_id'].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'user_merchant_received_counts'
    received_user_merchants = df[['User_id', 'Merchant_id']].join(frame)
    grouped = received_user_merchants.groupby(['User_id', 'Merchant_id'], as_index=False).sum()
    normalized_frame = min_max_normalize(received_user_merchants, frame.name)[frame.name]
    normalized_frame.name = 'user_merchant_received_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=['User_id', 'Merchant_id'], how='left')
    return df


coupon_received_features = [user_received_counts, user_received_coupon_counts, merchant_received_counts, merchant_received_coupon_counts, user_merchant_received_counts]

def user_merchants(df):
    #每个用户相关的商家数量(或领取了优惠券、或购买了东西)
    grouped = df[['User_id', 'Merchant_id']].groupby('User_id')['Merchant_id'].nunique().reset_index()
    normalized = min_max_normalize(grouped, 'Merchant_id').rename(columns={'Merchant_id': 'user_merchants'})
    df = df.merge(normalized, on='User_id', how='left')
    return df


def merchant_users(df):
    #每个商家相关的用户数量(或领取了优惠券、或购买了东西)
    grouped = df[['User_id', 'Merchant_id']].groupby('Merchant_id')['User_id'].nunique().reset_index()
    normalized = min_max_normalize(grouped, 'User_id').rename(columns={'User_id': 'merchant_users'})
    df = df.merge(normalized, on='Merchant_id', how='left')
    return df


user_merchant_dataset_features = [user_merchants, merchant_users]

def add_label_features(df):
    #加入预测区间的一些特征：距离、优惠券、商家。。。
    dataset_features = list()
    dataset_features.append(add_distance_rate)
    dataset_features.append(add_coupon_features)
    dataset_features.extend(coupon_received_features)
    dataset_features.extend(user_merchant_dataset_features)

    for f in dataset_features:
        df = f(df)

    df.fillna(-1, inplace=True)
    return df

def gen_feature_data(features_dir, dataset, output):
    #将之前特征区间提取的特征数据和预测区间的数据结合起来，保存在output里
    user_features = pd.read_csv(features_dir + 'user_features.csv').astype(str)
    merchant_features = pd.read_csv(features_dir + 'merchant_features.csv').astype(str)
    user_merchant_features = pd.read_csv(features_dir + 'user_merchant_features.csv').astype(str)

    columns = dataset.columns.tolist()
    df = dataset.merge(user_features, on='User_id', how='left')
    df = df.merge(merchant_features, on='Merchant_id', how='left')
    df = df.merge(user_merchant_features, on=['User_id', 'Merchant_id'], how='left')

    df = add_label_features(df)
    df.drop(columns, axis=1, inplace=True)
    df.fillna(np.nan, inplace=True)
    print(df.shape)
    print('start dump feature data')
    df.to_csv(output, index=False)

def get_time_diff(date_received, date_consumed):
    # 计算时间差
    month_diff = int(date_consumed[-4:-2]) - int(date_received[-4:-2])
    if month_diff == 0:
        return int(date_consumed[-2:]) - int(date_received[-2:])
    else:
        return int(date_consumed[-2:]) - int(date_received[-2:]) + month_diff * 30
    
def add_label(df):
    #生成target
    frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' and get_time_diff(z, y) <= 15 else 0., df['Coupon_id'], df['Date'], df['Date_received'])))
    frame.name = 'Label'
    print('pos_counts: {0}, neg_counts: {1}'.format(sum(frame == 1), sum(frame == 0)))
    df = df.join(frame)
    return df

def gen_label_data(dataset, output):
    df = add_label(dataset)
    print('start dump label data')
    df[['Label']].astype(float).to_csv(output, index=False)


def gen_data(feature_path,label_path, label=True):
    dataset = pd.read_csv(label_path).astype(str)
    gen_feature_data(feature_path, dataset, feature_path + 'train_features.csv')
    if label:
        gen_label_data(dataset, feature_path + 'labels.csv')


if __name__ == '__main__':
    print('generate train data...')
    gen_data(train_feature_save_path,offline_train_label_data_path)
    print('generate validate data...')
    gen_data(validate_feature_save_path,offline_validate_label_data_path)
    print('generate predict features...')
    gen_data(predict_feature_save_path, label=False)

    # generate train data...
    # (258446, 44)
    # start dump feature data
    # pos_counts: 23485, neg_counts: 234961
    # start dump label data
    # generate validate data...
    # (137167, 44)
    # start dump feature data
    # pos_counts: 9073, neg_counts: 128094
    # start dump label data
    # generate predict features...
    # (113640, 44)
    # start dump feature data

    # generate train data...
    # sort begin...
    # (258446, 92)
    # start dump feature data
    # pos_counts: 23485, neg_counts: 234961
    # start dump label data
    # generate validate data...
    # sort begin...
    # (137167, 92)
    # start dump feature data
    # pos_counts: 9073, neg_counts: 128094
    # start dump label data
    # generate predict features...
    # sort begin...
    # (113640, 92)
    # start dump feature data
