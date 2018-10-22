#!/usr/bin/env python
# encoding: utf-8

#划分数据集的时间段定义
train_feature_start_time = '20160201'          #训练集特征区间开始时间
train_feature_end_time = '20160514'          #训练集特征区间结束时间
train_label_start_time = '20160515'          #训练集预测区间开始时间
train_label_end_time = '20160615'          #训练集预测区间结束时间

validate_feature_start_time = '20160101'          #验证集特征区间开始时间
validate_feature_end_time = '20160413'          #验证集特征区间结束时间
validate_label_start_time = '20160414'          #验证集预测区间开始时间
validate_label_end_time = '20160514'          #验证集预测区间结束时间

predict_feature_start_time = '20160315'          #测试集特征区间开始时间
predict_feature_end_time = '20160630'          #测试集特征区间结束时间
predict_label_start_time = '20160701'          #测试集预测区间开始时间
predict_label_end_time = '20160731'          #测试集预测区间结束时间

#划分数据集后的存储路径
offline_train_feature_data_path='data/train/offline_train_feature_data.csv'
offline_train_label_data_path='data/train/offline_train_label_data.csv'
offline_validate_feature_data_path='data/validate/offline_validate_feature_data.csv'
offline_validate_label_data_path='data/validate/offline_validate_label_data.csv'
offline_predict_feature_data_path='data/predict/offline_predict_feature_data.csv'
offline_predict_label_data_path='data/predict/offline_predict_label_data.csv'

online_train_feature_data_path='data/train/online_train_feature_data.csv'
online_validate_feature_data_path='data/validate/online_validate_feature_data.csv'
online_predict_feature_data_path='data/predict/online_predict_feature_data.csv'

#特征提取后的存储路径
train_feature_save_path='data/train/'
validate_feature_save_path='data/validate/'
predict_feature_save_path='data/predict/'