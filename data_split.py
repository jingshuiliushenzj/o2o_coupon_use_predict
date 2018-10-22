#!/usr/bin/env python
# encoding: utf-8
import pandas as pd

from config import *
def get_df(file_path):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width',200)
    mylist = []
    for chunk in  pd.read_csv(file_path, chunksize=20000,keep_default_na=False):
        mylist.append(chunk)
    temp_df = pd.concat(mylist, axis= 0)
    del mylist
    return temp_df

#划分特征区间数据集,通过领取时间和消费时间
def filter_feature_data(data,start_time,end_time):
    return data[((data['Date'] > start_time) & (data['Date'] < end_time))|((data['Date']=='null')&(data['Date_received'] > start_time) & (data['Date_received'] < end_time))]

#划分预测区间数据集，根据领取时间
def filter_label_data(data,start_time,end_time):
    return data[(data['Date_received'] > start_time) & (data['Date_received'] < end_time)]
#划分线下数据集
def offline_data_split(origin):
    #训练集
    train_feature_data=filter_feature_data(origin,train_feature_start_time,train_feature_end_time)
    train_label_data=filter_label_data(origin,train_label_start_time,train_label_end_time)

    #验证集
    validate_feature_data = filter_feature_data(origin, validate_feature_start_time, validate_feature_end_time)
    validate_label_data = filter_label_data(origin, validate_label_start_time, validate_label_end_time)

    #测试集
    predict_feature_data = filter_feature_data(origin, predict_feature_start_time, predict_feature_end_time)
    predict_label_data = filter_label_data(origin, predict_label_start_time, predict_label_end_time)

    return train_feature_data,train_label_data,validate_feature_data,validate_label_data,predict_feature_data,predict_label_data


# 划分线上数据集
def online_data_split(origin):
    # 训练集
    train_feature_data = filter_feature_data(origin, train_feature_start_time, train_feature_end_time)

    # 验证集
    validate_feature_data = filter_feature_data(origin, validate_feature_start_time, validate_feature_end_time)

    # 测试集
    predict_feature_data = filter_feature_data(origin, predict_feature_start_time, predict_feature_end_time)

    return train_feature_data,  validate_feature_data,predict_feature_data

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    offline_data=get_df('data/ccf_offline_stage1_train.csv')
    offline_train_feature_data, offline_train_label_data, offline_validate_feature_data, offline_validate_label_data, offline_predict_feature_data, offline_predict_label_data=offline_data_split(offline_data)

    offline_train_feature_data.to_csv(offline_train_feature_data_path, index=False)
    offline_train_label_data.to_csv(offline_train_label_data_path, index=False)
    offline_validate_feature_data.to_csv(offline_validate_feature_data_path, index=False)
    offline_validate_label_data.to_csv(offline_validate_label_data_path, index=False)
    offline_predict_feature_data.to_csv(offline_predict_feature_data_path, index=False)
    offline_predict_label_data.to_csv(offline_predict_label_data_path, index=False)

    online_data = get_df('data/ccf_online_stage1_train.csv')

    active_user_online_record = online_data[online_data['User_id'].isin(offline_data['User_id'].tolist())]#只返回线下活跃的线上用户
    online_train_feature_data, online_validate_feature_data,  online_predict_feature_data = online_data_split(active_user_online_record)

    online_train_feature_data.to_csv(online_train_feature_data_path, index=False)
    online_validate_feature_data.to_csv(online_validate_feature_data_path, index=False)
    online_predict_feature_data.to_csv(online_predict_feature_data_path, index=False)
