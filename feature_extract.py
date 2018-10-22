#!/usr/bin/env python  
# encoding: utf-8
import pandas as pd

from config import *

def min_max_normalize(df, name):
    # 归一化
    max_number = df[name].max()
    min_number = df[name].min()
    # assert max_number != min_number, 'max == min in COLUMN {0}'.format(name)
    df[name] = df[name].map(lambda x: float(x - min_number + 1) / float(max_number - min_number + 1))
    # 做简单的平滑,试试效果如何
    return df
def user_normal_consume_rate(df, online=False):
    # 用户无优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y == 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'user_normal_consume_rate' if not online else 'online_user_normal_consume_rate'
    normal_consume_user = df[['User_id']].join(frame)
    grouped = normal_consume_user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_normal_consume_normalized_rate' if not online else 'online_user_normal_consume_normalized_rate'
    ss=grouped[frame.name].mean().join(normalized_frame)
    return ss

def user_none_consume_rate(df, online=False):
    # 用户获得优惠券但没有消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x == 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'user_none_consume_rate' if not online else 'online_user_none_consume_rate'
    none_consume_user = df[['User_id']].join(frame)
    grouped = none_consume_user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_none_consume_normalized_rate' if not online else 'online_user_none_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_coupon_consume_rate(df, online=False):
    # 用户优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'user_coupon_consume_rate' if not online else 'online_user_coupon_consume_rate'
    coupon_consume_user = df[['User_id']].join(frame)
    grouped = coupon_consume_user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_consume_normalized_rate' if not online else 'online_user_coupon_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_consume_coupon_rate(df, online=False):
    # 用户领取优惠券后进行核销率
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'user_consume_coupon_counts'
    coupon_consume_user = df[['User_id']].join(frame)
    frame = df['Coupon_id'].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'user_received_coupon_counts'
    coupon_consume_user = coupon_consume_user.join(frame)
    grouped = coupon_consume_user.groupby('User_id', as_index=False).sum()
    frame = pd.Series(list(map(lambda x, y: -1 if x == 0 else float(y) / float(x), grouped['user_received_coupon_counts'], grouped['user_consume_coupon_counts'])))
    frame.name = 'user_consume_coupon_rate' if online else 'online_user_consume_coupon_rate'
    ss=grouped[['User_id']].join(frame)
    return ss

#提取用户特征的函数列表
user_consume_rates = [user_normal_consume_rate, user_none_consume_rate, user_coupon_consume_rate, user_consume_coupon_rate]

def online_user_action_0_rate(df, online=True):
    assert online, 'No action feature in offline data'
    frame = df['Action'].map(lambda x: 1. if str(x) == '0' else 0.)
    frame.name = 'online_user_action_0_rate'
    user = df[['User_id']].join(frame)
    grouped = user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'online_user_action_0_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def online_user_action_1_rate(df, online=True):
    assert online, 'No action feature in offline data'
    frame = df['Action'].map(lambda x: 1. if str(x) == '1' else 0.)
    frame.name = 'online_user_action_1_rate'
    user = df[['User_id']].join(frame)
    grouped = user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'online_user_action_1_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def online_user_action_2_rate(df, online=True):
    assert online, 'No action feature in offline data'
    frame = df['Action'].map(lambda x: 1. if str(x) == '2' else 0.)
    frame.name = 'online_user_action_2_rate'
    user = df[['User_id']].join(frame)
    grouped = user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'online_user_action_2_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


online_user_action_types = [online_user_action_0_rate, online_user_action_1_rate, online_user_action_2_rate]

def user_fixed_discount_rate(df, online=False):
    # 用户优惠券消费中限时低价消费率及次数归一化
    assert online, 'no fixed offline discount '
    mask = pd.Series(list(map(lambda x, y, z: True if x != 'null' and y != 'null' and z != 'null' else False, df['Date'], df['Coupon_id'], df['Discount_rate'])))
    discount_users = df[mask]
    frame = discount_users['Discount_rate'].map(lambda x: 1. if str(x) == 'fixed' else 0.)
    frame.name = 'user_fixed_discount_rate' if not online else 'online_user_fixed_discount_rate'
    discounts = discount_users.join(frame)[['User_id', frame.name]]
    grouped = discounts.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_fixed_discount_normalized_rate' if not online else 'online_user_fixed_discount_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)

def user_direct_discount_rate(df, online=False):
    # 用户优惠券消费中直接折扣消费(非满减)率及次数归一化
    mask = pd.Series(list(map(lambda x, y, z: True if x != 'null' and y != 'null' and z != 'null' else False, df['Date'], df['Coupon_id'], df['Discount_rate'])))
    discount_users = df[mask]
    frame = discount_users['Discount_rate'].map(lambda x: 1. if str(x).find(':') == -1 else 0.)
    frame.name = 'user_direct_discount_rate' if not online else 'online_user_direct_discount_rate'
    discounts = discount_users.join(frame)[['User_id', frame.name]]
    grouped = discounts.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_direct_discount_normalized_rate' if not online else 'online_user_direct_discount_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)

def user_consume_time_rate(df, online=False):
    # 用户获得优惠券后到使用消费券之间的平均等待时间计算率
    valid_time_user = df.loc[df['Date'] != 'null'].loc[df['Date_received'] != 'null']
    date_consumed = valid_time_user['Date'].map(lambda x: int(x[-4:-2]) * 30 + int(x[-2:]))
    date_received = valid_time_user['Date_received'].map(lambda x: int(x[-4:-2]) * 30 + int(x[-2:]))
    frame = pd.Series(list(map(lambda x, y: 1. - float(x - y) / 15 if x - y < 15 else 0., date_consumed, date_received)), index=valid_time_user.index)
    frame.name = 'user_consume_time_rate' if not online else 'online_user_consume_time_rate'
    valid_time_user = valid_time_user[['User_id']].join(frame)
    return valid_time_user.groupby('User_id', as_index=False)[frame.name].mean()

def user_consume_merchants(df, online=False):
    # 用户优惠券消费过的不同商家数量归一化
    mask = pd.Series(list(map(lambda x, y: True if x != 'null' and y != 'null' else False, df['Date'], df['Coupon_id'])))
    grouped = df[mask][['User_id', 'Merchant_id']].groupby('User_id')['Merchant_id'].nunique().reset_index()
    ss=min_max_normalize(grouped, 'Merchant_id').rename(columns={'Merchant_id': 'user_consume_merchants_rate' if not online else 'online_user_consume_merchants_rate'})
    return ss

def user_consume_coupons(df, online=False):
    # 用户优惠券消费过的不同优惠券数量归一化
    mask = pd.Series(list(map(lambda x, y: True if x != 'null' and y != 'null' else False, df['Date'], df['Coupon_id'])))
    grouped = df[mask][['User_id', 'Coupon_id']].groupby('User_id')['Coupon_id'].nunique().reset_index()
    return min_max_normalize(grouped, 'Coupon_id').rename(columns={'Coupon_id': 'user_consume_coupons_rate' if not online else 'online_user_consume_coupons_rate'})


#提取用户特征
def extract_user_feature(data,online=True):
    user_features = []
    user_features.extend(user_consume_rates)#用户领取优惠券消费、用户直接消费、用户领取优惠券但没有消费、用户优惠券的核销率
    if online:
        user_features.extend(online_user_action_types)#用户线上点击率（三种类型的）
        user_features.append(user_fixed_discount_rate)#限时低价消费
    user_features.append(user_direct_discount_rate)#直接折扣消费
    user_features.append(user_consume_time_rate)#用户获得优惠券后到使用消费券之间的平均等待时间计算率
    user_features.append(user_consume_merchants)#用户优惠券消费过的不同商家数量
    user_features.append(user_consume_coupons) # 用户优惠券消费过的不同优惠券数量

    user_feature_data = data[['User_id']].drop_duplicates(['User_id'])

    for f in user_features:
        user_feature_data = user_feature_data.merge(f(data, online=online), on='User_id', how='left')
    user_feature_data.fillna(-1, inplace=True)

    return user_feature_data

def discount_floor_partition(x):
    # 满减下限分类
    x = str(x)
    if x == 'null':
        return -1
    # fixed暂记为1,其实应该也算缺失值
    elif x == 'fixed':
        return 1
    # 固定折扣率是对任意消费额度都有折扣,所以考虑2,3,4,5各加一次
    elif x.find(':') == -1:
        if 0 <= float(x) <= 1:
            return 0
        else:
            print('Discount Rate Error: %s' % x)
            raise
    discount_floor = int(x.split(':')[0])
    if 0 <= discount_floor < 50:
        return 2
    elif discount_floor < 200:
        return 3
    elif discount_floor < 500:
        return 4
    else:
        return 5

def discount_rate_calculation(x):
    # 计算折率
    x = str(x)
    if x == 'null' or x == 'fixed':
        return -1
    elif x.find(':') == -1:
        return float(x)
    else:
        nums = x.split(':')
        return float(nums[1]) / float(nums[0])

def user_coupon_discount_floor_50_rate(df, online=False):
    # 用户优惠券消费中满0~50折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' and (z == 2 or z == 0) else 0.,
                               df['Date'], df['Coupon_id'], df['discount_floor_partition'])))
    frame.name = 'user_coupon_discount_floor_50_rate' if not online else 'online_user_coupon_discount_floor_50_rate'
    coupon_discount_user = df[['User_id']].join(frame)
    grouped = coupon_discount_user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_discount_floor_50_normalized_rate' if not online else 'online_user_coupon_discount_floor_50_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)

def user_coupon_discount_floor_200_rate(df, online=False):
    # 用户优惠券消费中满50~200折扣次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' and (z == 3 or z == 0) else 0.,
                               df['Date'], df['Coupon_id'], df['discount_floor_partition'])))
    frame.name = 'user_coupon_discount_floor_200_rate' if not online else 'online_user_coupon_discount_floor_200_rate'
    coupon_discount_user = df[['User_id']].join(frame)
    grouped = coupon_discount_user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_discount_floor_200_normalized_rate' if not online else 'online_user_coupon_discount_floor_200_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)

def user_coupon_discount_floor_500_rate(df, online=False):
    # 用户优惠券消费中满200~500折扣次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' and (z == 4 or z == 0) else 0.,
                               df['Date'], df['Coupon_id'], df['discount_floor_partition'])))
    frame.name = 'user_coupon_discount_floor_500_rate' if not online else 'online_user_coupon_discount_floor_500_rate'
    coupon_discount_user = df[['User_id']].join(frame)
    grouped = coupon_discount_user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_discount_floor_500_normalized_rate' if not online else 'online_user_coupon_discount_floor_500_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)

def user_coupon_discount_floor_others_rate(df, online=False):
    # 用户优惠券消费中其他满减次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' and (z == 5 or z == 0) else 0.,
                               df['Date'], df['Coupon_id'], df['discount_floor_partition'])))
    frame.name = 'user_coupon_discount_floor_others_rate' if not online else 'online_user_coupon_discount_floor_others_rate'
    coupon_discount_user = df[['User_id']].join(frame)
    grouped = coupon_discount_user.groupby('User_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_discount_floor_others_normalized_rate' if not online else 'online_user_coupon_discount_floor_others_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)

user_coupon_discount_floor_rates = [user_coupon_discount_floor_50_rate, user_coupon_discount_floor_200_rate,
                                        user_coupon_discount_floor_500_rate, user_coupon_discount_floor_others_rate]

def user_average_discount_rate(df, online=False):
    # 用户优惠券消费平均消费折率
    mask = pd.Series(list(map(lambda x, y, z: True if x != 'null' and y != 'null' and z != -1 else False, df['Date'], df['Coupon_id'], df['discount_rate'])))
    discount_rates = df[mask][['User_id', 'discount_rate']]
    discount_rates['discount_rate'] = discount_rates['discount_rate'].astype(float)
    grouped = discount_rates.groupby('User_id', as_index=False)
    return grouped['discount_rate'].mean().rename(columns={'discount_rate': 'user_discount_average_rate' if not online else 'online_user_discount_average_rate'})

def add_user_coupon_features(df, online=False):
    frame = df['Discount_rate'].map(discount_floor_partition)
    frame.name = 'discount_floor_partition'
    df = df.join(frame)
    frame = df['Discount_rate'].map(discount_rate_calculation)
    frame.name = 'discount_rate'
    df = df.join(frame)

    user_coupon_features = []
    user_coupon_features.extend(user_coupon_discount_floor_rates)
    user_coupon_features.append(user_average_discount_rate)

    user_coupon_feature_data = df[['User_id']].drop_duplicates(['User_id'])

    for f in user_coupon_features:
        user_coupon_feature_data = user_coupon_feature_data.merge(f(df, online=online), on='User_id', how='left')
    user_coupon_feature_data.fillna(-1, inplace=True)

    return user_coupon_feature_data

def merchant_normal_consume_rate(df):
    # 商家无优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y == 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'merchant_normal_consume_rate'
    normal_consume_merchant = df[['Merchant_id']].join(frame)
    grouped = normal_consume_merchant.groupby('Merchant_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_normal_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_none_consume_rate(df):
    # 商家获得优惠券但没有消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x == 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'merchant_none_consume_rate'
    none_consume_merchant = df[['Merchant_id']].join(frame)
    grouped = none_consume_merchant.groupby('Merchant_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_none_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_coupon_consume_rate(df):
    # 商家优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'merchant_coupon_consume_rate'
    coupon_consume_merchant = df[['Merchant_id']].join(frame)
    grouped = coupon_consume_merchant.groupby('Merchant_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_consume_coupon_rate(df):
    # 商家的优惠券被领取后的核销率
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'merchant_consume_coupon_counts'
    coupon_consume_merchant = df[['Merchant_id']].join(frame)
    frame = df['Coupon_id'].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'merchant_received_coupon_counts'
    coupon_consume_merchant = coupon_consume_merchant.join(frame)
    grouped = coupon_consume_merchant.groupby('Merchant_id', as_index=False).sum()
    frame = pd.Series(list(map(lambda x, y: -1 if x == 0 else float(y) / float(x), grouped['merchant_received_coupon_counts'], grouped['merchant_consume_coupon_counts'])))
    frame.name = 'merchant_consume_coupon_rate'
    return grouped[['Merchant_id']].join(frame)


merchant_consume_rates = [merchant_normal_consume_rate, merchant_none_consume_rate, merchant_coupon_consume_rate, merchant_consume_coupon_rate]

def merchant_direct_discount_rate(df):
    # 商家优惠券消费中直接折扣消费(非满减)率及次数归一化
    mask = pd.Series(list(map(lambda x, y, z: True if x != 'null' and y != 'null' and z != 'null' else False, df['Date'], df['Coupon_id'], df['Discount_rate'])))
    discount_merchants = df[mask]
    frame = discount_merchants['Discount_rate'].map(lambda x: 1. if str(x).find(':') == -1 else 0.)
    frame.name = 'merchant_direct_discount_rate'
    discounts = discount_merchants.join(frame)[['Merchant_id', frame.name]]
    grouped = discounts.groupby('Merchant_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_direct_discount_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)

def merchant_consume_time_rate(df):
    # 商家被消费的优惠券从被用户获得到使用之间的时间计算率
    valid_time_merchant = df.loc[df['Date'] != 'null'].loc[df['Date_received'] != 'null']
    date_consumed = valid_time_merchant['Date'].map(lambda x: int(x[-4:-2]) * 30 + int(x[-2:]))
    date_received = valid_time_merchant['Date_received'].map(lambda x: int(x[-4:-2]) * 30 + int(x[-2:]))
    frame = pd.Series(list(map(lambda x, y: 1. - float(x - y) / 15 if x - y < 15 else 0., date_consumed, date_received)), index=valid_time_merchant.index)
    frame.name = 'merchant_consume_time_rate'
    valid_time_merchant = valid_time_merchant[['Merchant_id']].join(frame)
    return valid_time_merchant.groupby('Merchant_id', as_index=False)[frame.name].mean()


def merchant_consume_users(df):
    # 消费过商家优惠券的不同用户数量归一化
    mask = pd.Series(list(map(lambda x, y: True if x != 'null' and y != 'null' else False, df['Date'], df['Coupon_id'])))
    grouped = df[mask][['User_id', 'Merchant_id']].groupby('Merchant_id')['User_id'].nunique().reset_index()
    return min_max_normalize(grouped, 'User_id').rename(columns={'User_id': 'merchant_consume_users_rate'})


def merchant_consume_coupons(df):
    # 商家被消费过的不同优惠券数量归一化
    mask = pd.Series(list(map(lambda x, y: True if x != 'null' and y != 'null' else False, df['Date'], df['Coupon_id'])))
    grouped = df[mask][['Coupon_id', 'Merchant_id']].groupby('Merchant_id')['Coupon_id'].nunique().reset_index()
    return min_max_normalize(grouped, 'Coupon_id').rename(columns={'Coupon_id': 'merchant_consume_coupons_rate'})

def add_merchant_features(df):
    merchant_features = []
    merchant_features.extend(merchant_consume_rates)
    merchant_features.append(merchant_direct_discount_rate)
    merchant_features.append(merchant_consume_time_rate)
    merchant_features.append(merchant_consume_users)
    merchant_features.append(merchant_consume_coupons)

    merchant_feature_data = df[['Merchant_id']].drop_duplicates(['Merchant_id'])

    for f in merchant_features:
        merchant_feature_data = merchant_feature_data.merge(f(df), on='Merchant_id', how='left')
    merchant_feature_data.fillna(-1, inplace=True)

    return merchant_feature_data

def merchant_coupon_discount_floor_50_rate(df):
    # 用户正常消费中满0~50折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' and (z == 2 or z == 0) else 0., df['Date'], df['Coupon_id'], df['discount_floor_partition'])))
    frame.name = 'merchant_coupon_discount_floor_50_rate'
    coupon_discount_merchant = df[['Merchant_id']].join(frame)
    grouped = coupon_discount_merchant.groupby('Merchant_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_discount_floor_50_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_coupon_discount_floor_200_rate(df):
    # 用户正常消费中满50~200折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' and (z == 3 or z == 0) else 0., df['Date'], df['Coupon_id'], df['discount_floor_partition'])))
    frame.name = 'merchant_coupon_discount_floor_200_rate'
    coupon_discount_merchant = df[['Merchant_id']].join(frame)
    grouped = coupon_discount_merchant.groupby('Merchant_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_discount_floor_200_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_coupon_discount_floor_500_rate(df):
    # 用户正常消费中满200~500折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' and (z == 4 or z == 0) else 0., df['Date'], df['Coupon_id'], df['discount_floor_partition'])))
    frame.name = 'merchant_coupon_discount_floor_500_rate'
    coupon_discount_merchant = df[['Merchant_id']].join(frame)
    grouped = coupon_discount_merchant.groupby('Merchant_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_discount_floor_500_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_coupon_discount_floor_others_rate(df):
    # 用户正常消费中其他折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' and (z == 5 or z == 0) else 0., df['Date'], df['Coupon_id'], df['discount_floor_partition'])))
    frame.name = 'merchant_coupon_discount_floor_others_rate'
    coupon_discount_merchant = df[['Merchant_id']].join(frame)
    grouped = coupon_discount_merchant.groupby('Merchant_id', as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_discount_floor_others_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)

merchant_coupon_discount_floor_rates = [merchant_coupon_discount_floor_50_rate, merchant_coupon_discount_floor_200_rate, merchant_coupon_discount_floor_500_rate, merchant_coupon_discount_floor_others_rate]

def merchant_average_discount_rate(df):
    # 商家优惠券消费平均消费折率
    mask = pd.Series(list(map(lambda x, y, z: True if x != 'null' and y != 'null' and z != 'null' else False, df['Date'], df['Coupon_id'], df['discount_rate'])))
    discount_rates = df[mask][['Merchant_id', 'discount_rate']]
    discount_rates['discount_rate'] = discount_rates['discount_rate'].astype(float)
    grouped = discount_rates.groupby('Merchant_id', as_index=False)
    return grouped['discount_rate'].mean().rename(columns={'discount_rate': 'merchant_discount_average_rate'})
def add_merchant_coupon_features(df):
    frame = df['Discount_rate'].map(discount_floor_partition)
    frame.name = 'discount_floor_partition'
    df = df.join(frame)
    frame = df['Discount_rate'].map(discount_rate_calculation)
    frame.name = 'discount_rate'
    df = df.join(frame)

    merchant_coupon_features = []
    merchant_coupon_features.extend(merchant_coupon_discount_floor_rates)
    merchant_coupon_features.append(merchant_average_discount_rate)

    merchant_coupon_feature_data = df[['Merchant_id']].drop_duplicates(['Merchant_id'])

    for f in merchant_coupon_features:
        merchant_coupon_feature_data = merchant_coupon_feature_data.merge(f(df), on='Merchant_id', how='left')
    merchant_coupon_feature_data.fillna(-1, inplace=True)

    return merchant_coupon_feature_data
def user_merchant_normal_consume_rate(df):
    # 用户对商家的所有消费中,普通消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y == 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'user_merchant_normal_consume_rate'
    normal_consume_user_merchant = df[['User_id', 'Merchant_id']].join(frame)
    grouped = normal_consume_user_merchant.groupby(['User_id', 'Merchant_id'], as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_merchant_normal_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_merchant_none_consume_rate(df):
    # 用户对商家的所有消费中,不消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x == 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'user_merchant_none_consume_rate'
    none_consume_user_merchant = df[['User_id', 'Merchant_id']].join(frame)
    grouped = none_consume_user_merchant.groupby(['User_id', 'Merchant_id'], as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_merchant_none_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_merchant_coupon_consume_rate(df):
    # 用户对商家的所有消费中,优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y == 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'user_merchant_coupon_consume_rate'
    coupon_consume_user_merchant = df[['User_id', 'Merchant_id']].join(frame)
    grouped = coupon_consume_user_merchant.groupby(['User_id', 'Merchant_id'], as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_merchant_coupon_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_merchant_consume_coupon_rate(df):
    # 用户领取商家的优惠券后的核销率
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'user_merchant_consume_coupon_counts'
    coupon_consume_merchant = df[['User_id', 'Merchant_id']].join(frame)
    frame = df['Coupon_id'].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'user_merchant_received_coupon_counts'
    coupon_consume_merchant = coupon_consume_merchant.join(frame)
    grouped = coupon_consume_merchant.groupby(['User_id', 'Merchant_id'], as_index=False).sum()
    frame = pd.Series(list(map(lambda x, y: -1 if x == 0 else float(y) / float(x), grouped['user_merchant_received_coupon_counts'], grouped['user_merchant_consume_coupon_counts'])))
    frame.name = 'user_merchant_consume_coupon_rate'
    return grouped[['User_id', 'Merchant_id']].join(frame)

user_merchant_consume_rate = [user_merchant_normal_consume_rate, user_merchant_none_consume_rate, user_merchant_coupon_consume_rate, user_merchant_consume_coupon_rate]

def user_normal_consume_merchant_rate(df):
    # 用户对每个商家的普通消费次数占用户普通消费所有商家的比重
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y == 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'normal_consume'
    normal_consume_user_merchant = df[['User_id', 'Merchant_id']].join(frame)
    user_dicts = dict(normal_consume_user_merchant.groupby('User_id')[frame.name].sum())
    user_merchant_dicts = dict(normal_consume_user_merchant.groupby(['User_id', 'Merchant_id'])[frame.name].sum())
    unique_user_merchant = df[['User_id', 'Merchant_id']].drop_duplicates(['User_id', 'Merchant_id'])
    frame = pd.Series(list(map(lambda x, y: user_merchant_dicts[(x, y)] / user_dicts[x], unique_user_merchant['User_id'], unique_user_merchant['Merchant_id'])))
    frame.name = 'user_normal_consume_merchant_rate'
    return unique_user_merchant.join(frame)


def user_none_consume_merchant_rate(df):
    # 用户对每个商家的不消费次数占用户不消费的所有商家的比重
    frame = pd.Series(list(map(lambda x, y: 1. if x == 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'none_consume'
    none_consume_user_merchant = df[['User_id', 'Merchant_id']].join(frame)
    user_dicts = dict(none_consume_user_merchant.groupby('User_id')[frame.name].sum())
    user_merchant_dicts = dict(none_consume_user_merchant.groupby(['User_id', 'Merchant_id'])[frame.name].sum())
    unique_user_merchant = df[['User_id', 'Merchant_id']].drop_duplicates(['User_id', 'Merchant_id'])
    frame = pd.Series(list(map(lambda x, y: user_merchant_dicts[(x, y)] / user_dicts[x], unique_user_merchant['User_id'], unique_user_merchant['Merchant_id'])))
    frame.name = 'user_none_consume_merchant_rate'
    return unique_user_merchant.join(frame)


def user_coupon_consume_merchant_rate(df):
    # 用户对每个商家的优惠券消费次数占用户优惠券消费所有商家的比重
    frame = pd.Series(list(map(lambda x, y: 1. if x != 'null' and y != 'null' else 0., df['Date'], df['Coupon_id'])))
    frame.name = 'coupon_consume'
    coupon_consume_user_merchant = df[['User_id', 'Merchant_id']].join(frame)
    user_dicts = dict(coupon_consume_user_merchant.groupby('User_id')[frame.name].sum())
    user_merchant_dicts = dict(coupon_consume_user_merchant.groupby(['User_id', 'Merchant_id'])[frame.name].sum())
    unique_user_merchant = df[['User_id', 'Merchant_id']].drop_duplicates(['User_id', 'Merchant_id'])
    frame = pd.Series(list(map(lambda x, y: user_merchant_dicts[(x, y)] / user_dicts[x], unique_user_merchant['User_id'], unique_user_merchant['Merchant_id'])))
    frame.name = 'user_coupon_consume_merchant_rate'
    return unique_user_merchant.join(frame)
user_consume_merchant_rate = [user_normal_consume_merchant_rate, user_none_consume_merchant_rate, user_coupon_consume_merchant_rate]

def add_user_merchant_features(df):
    user_merchant_features = []
    user_merchant_features.extend(user_merchant_consume_rate)
    user_merchant_features.extend(user_consume_merchant_rate)

    user_merchant_feature_data = df[['User_id', 'Merchant_id']].drop_duplicates(['User_id', 'Merchant_id'])

    for f in user_merchant_features:
        user_merchant_feature_data = user_merchant_feature_data.merge(f(df), on=['User_id', 'Merchant_id'], how='left')
    user_merchant_feature_data.fillna(-1, inplace=True)

    return user_merchant_feature_data

def add_user_features(df, online=False):
    user_features = []
    user_features.extend(user_consume_rates)
    if online:
        user_features.extend(online_user_action_types)
        user_features.append(user_fixed_discount_rate)
    user_features.append(user_direct_discount_rate)
    user_features.append(user_consume_time_rate)
    user_features.append(user_consume_merchants)
    user_features.append(user_consume_coupons)

    user_feature_data = df[['User_id']].drop_duplicates(['User_id'])

    for f in user_features:
        user_feature_data = user_feature_data.merge(f(df, online=online), on='User_id', how='left')
    user_feature_data.fillna(-1, inplace=True)

    return user_feature_data

def add_offline_online_features(offline_data, active_online_data):
    active_normal_consume = active_online_data.loc[active_online_data['Date'] != 'null'].loc[active_online_data['Coupon_id'] == 'null']
    active_none_consume = active_online_data.loc[active_online_data['Date'] == 'null'].loc[active_online_data['Coupon_id'] != 'null']
    active_coupon_consume = active_online_data.loc[active_online_data['Date'] != 'null'].loc[active_online_data['Coupon_id'] != 'null']
    new_active_data = active_online_data[['User_id']].drop_duplicates()
    new_active_data = add_count_users(new_active_data, active_normal_consume[['User_id']], 'online_normal_consume_count', online=True)
    new_active_data = add_count_users(new_active_data, active_none_consume[['User_id']], 'online_none_consume_count', online=True)
    new_active_data = add_count_users(new_active_data, active_coupon_consume[['User_id']], 'online_coupon_consume_count', online=True)
    new_active_data = add_count_users(new_active_data, active_online_data[['User_id']], 'online_count', online=True)

    offline_normal_consume = offline_data.loc[offline_data['Date'] != 'null'].loc[offline_data['Coupon_id'] == 'null']
    offline_none_consume = offline_data.loc[offline_data['Date'] == 'null'].loc[offline_data['Coupon_id'] != 'null']
    offline_coupon_consume = offline_data.loc[offline_data['Date'] != 'null'].loc[offline_data['Coupon_id'] != 'null']
    new_offline_data = offline_data[['User_id']].drop_duplicates()
    new_offline_data = add_count_users(new_offline_data, offline_normal_consume[['User_id']], 'offline_normal_consume_count')
    new_offline_data = add_count_users(new_offline_data, offline_none_consume[['User_id']], 'offline_none_consume_count')
    new_offline_data = add_count_users(new_offline_data, offline_coupon_consume[['User_id']], 'offline_coupon_consume_count')
    new_offline_data = add_count_users(new_offline_data, offline_data, 'offline_count')

    user_online_feature_data = new_offline_data.merge(new_active_data, on='User_id', how='left')
    user_online_feature_data.fillna(0, inplace=True)
    user_online_feature_data = user_online_feature_data.join(calc_rate(user_online_feature_data['offline_normal_consume_count'], user_online_feature_data['online_normal_consume_count'], 'offline_normal_consume_rate'))
    user_online_feature_data = user_online_feature_data.join(calc_rate(user_online_feature_data['offline_none_consume_count'], user_online_feature_data['online_none_consume_count'], 'offline_none_consume_rate'))
    user_online_feature_data = user_online_feature_data.join(calc_rate(user_online_feature_data['offline_coupon_consume_count'], user_online_feature_data['online_coupon_consume_count'], 'offline_coupon_consume_rate'))
    user_online_feature_data = user_online_feature_data.join(calc_rate(user_online_feature_data['offline_count'], user_online_feature_data['online_count'], 'offline_rate'))
    user_online_feature_data = user_online_feature_data[['User_id', 'online_normal_consume_count_normalized', 'online_none_consume_count_normalized', 'online_coupon_consume_count_normalized', 'online_count_normalized', 'offline_normal_consume_rate', 'offline_none_consume_rate', 'offline_coupon_consume_rate', 'offline_rate']]

    return user_online_feature_data

def add_count_users(df, data, name, online=False):
    # 人数统计
    rlp = dict(data['User_id'].value_counts())
    print("df.head():",df.head())
    print("rlp.head",rlp)
    frame = df['User_id'].map(rlp)
    print("frame.head",frame.head())
    frame.name = name
    df = df.join(frame)

    if online:
        max_number = frame.max()
        min_number = frame.min()
        frame = frame.map(lambda x: float(x - min_number) / float(max_number - min_number))
        frame.name = '%s_normalized' % name
        df = df.join(frame)

    return df

def calc_rate(offline_column, online_column, name):
    frame = pd.Series(list(map(lambda x, y: (float(x) / (float(x) + float(y))) if float(x) + float(y) != 0 else -1, offline_column, online_column)))
    frame.name = name
    return frame


#特征提取
def extract_feature(offline_feature_data,online_feature_data,feature_save_path):
    print('start extract user features')
    user_feature_data = extract_user_feature(offline_feature_data,online=False)
    print('start extract user coupon features')
    user_coupon_feature_data = add_user_coupon_features(offline_feature_data)

    user_features = user_feature_data.merge(user_coupon_feature_data, on='User_id', how='outer')

    print('start extract merchant features')
    merchant_feature_data = add_merchant_features(offline_feature_data)
    print('start extract merchant coupon features')
    merchant_coupon_feature_data = add_merchant_coupon_features(offline_feature_data)

    merchant_features = merchant_feature_data.merge(merchant_coupon_feature_data, on='Merchant_id', how='outer')

    print('start extract user merchant features')
    user_merchant_features = add_user_merchant_features(offline_feature_data)

    print('start extract user online features')
    online_user_feature_data = add_user_features(online_feature_data, online=True)
    print('start extract user online coupon features')
    online_user_coupon_feature_data = add_user_coupon_features(online_feature_data, online=True)

    print('start extract user offline-online features')
    user_offline_online_feature_data = add_offline_online_features(offline_feature_data, online_feature_data)

    user_features = user_features.merge(online_user_feature_data, on='User_id', how='outer')
    user_features = user_features.merge(online_user_coupon_feature_data, on='User_id', how='outer')
    user_features = user_features.merge(user_offline_online_feature_data, on='User_id', how='outer')

    print('fill nan with -1')
    #user_features.fillna(-1, inplace=True)
    #merchant_features.fillna(-1, inplace=True)
    #user_merchant_features.fillna(-1, inplace=True)

    print('start dump feature data')
    #user_features.to_csv(feature_save_path + 'user_features.csv', index=False)
    #merchant_features.to_csv(feature_save_path + 'merchant_features.csv', index=False)
    #user_merchant_features.to_csv(feature_save_path + 'user_merchant_features.csv', index=False)
    return user_features

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    offline_train_feature_data=pd.read_csv(offline_train_feature_data_path,keep_default_na=False)
    offline_train_label_data = pd.read_csv(offline_train_label_data_path, keep_default_na=False)
    offline_validate_feature_data = pd.read_csv(offline_validate_feature_data_path, keep_default_na=False)
    offline_validate_label_data = pd.read_csv(offline_validate_label_data_path, keep_default_na=False)
    offline_predict_feature_data = pd.read_csv(offline_predict_feature_data_path, keep_default_na=False)
    offline_predict_label_data = pd.read_csv(offline_predict_label_data_path, keep_default_na=False)

    online_train_feature_data = pd.read_csv(online_train_feature_data_path, keep_default_na=False)
    online_validate_feature_data = pd.read_csv(online_validate_feature_data_path, keep_default_na=False)
    online_predict_feature_data = pd.read_csv(online_predict_feature_data_path, keep_default_na=False)

    extract_feature(offline_train_feature_data, online_train_feature_data,train_feature_save_path)
    #extract_feature(offline_validate_feature_data, online_validate_feature_data,validate_feature_save_path)
    #extract_feature(offline_predict_feature_data, online_predict_feature_data,predict_feature_save_path)
