#_*_coding: utf-8_*_
# 作者     : fengyao
# 创建时间 ：2018/9/12 12:04
# 文件     ：knn_titanic.py
#IDE       : PyCharm

import numpy as np
import operator

def file2matrix(filename, n):
    '''
    获取文件中的数据
    :param filename: 文件路径
    :param n: 特征数
    :return: 训练集和标签矩阵
    '''
    data = np.loadtxt(filename, delimiter=',')
    return_mat = data[:, : n].reshape((len(data), n))
    class_labels_vector = data[:, -1:].reshape((len(data), 1))
    return return_mat, class_labels_vector

def auto_norm(return_mat):
    '''
    特征缩放，归一化特征值
    :param return_mat: 数据集矩阵
    :return: 归一化矩阵
    '''
    min_values = return_mat.min(0)
    max_values = return_mat.max(0)
    ranges = max_values - min_values
    m = return_mat.shape[0]
    norm_data = return_mat - np.tile(min_values, (m, 1))
    norm_data = norm_data / np.tile(ranges, (m, 1))
    return norm_data, ranges, min_values

def classfy1(auto_train_data, train_labels, example, k):
    '''
    分类器
    :param train_data: 训练集特征矩阵
    :param labels: 训练集标签
    :param example: 测试集特征矩阵
    :param k: 统计参数
    :return: 预测值
    '''
    data_train_row = auto_train_data.shape[0]
    diff_mat = np.tile(example, (data_train_row, 1)) - train_data
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distance = sq_distance ** 0.5
    distance_labels = np.column_stack((distance, train_labels))
    sort_distance_increase = distance_labels[np.lexsort(distance_labels.T[0, None])]
    result = sort_distance_increase[:, 1:]
    # print(result)
    class_count = {}
    for i in range(k):
        vote_label = result[i][0]
        # print(vote_label)
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
        # print(class_count)
    # print(class_count)
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # print(sort_class_count)
    return sort_class_count[0][0]

def test(auto_train_data, train_labels, auto_test_data, test_labels):
    m_test = auto_test_data.shape[0]
    # test_result_list = []
    # test_result_mitrix = np.zeros((m_test, k_range))
    # error_rate_dict = {}
    # for k in range(6, 15):
    error_count = 0.0
    for i in range(m_test):
        example = auto_test_data[i]
        example_result = classfy1(auto_train_data, train_labels, example, 5)
        #print('预测结果为：%d, 实际结果为：%d' % (example_result, test_labels[i]))
        # test_result_list.append(example_result)
        if example_result != test_labels[i]:
            error_count += 1
    error_rate = error_count / float(m_test)
    print(error_rate)
        #print('错误率：', error_rate)
        # np.savetxt('test_result.csv', test_result, delimiter=',')
        # error_rate_dict[k] = error_rate
    # sort_error_rate = sorted(error_rate_dict.items(), key=operator.itemgetter(1))
    # print(sort_error_rate[0])

n = 8
k_range = 150
train_file = 'train.csv'
test_file = 'test.csv'

train_data, train_labels = file2matrix(train_file, n)
test_data, test_labels = file2matrix(test_file, n)

auto_train_data, train_ranges, train_min_values = auto_norm(train_data)
auto_test_data, test_ranges, test_min_values = auto_norm(test_data)
print(auto_train_data)
print(auto_test_data)


# example = auto_test_data[0]
# example_result = classfy1(train_data, train_labels, example, 10)
# print(example_result)
test(auto_train_data, train_labels, auto_test_data, test_labels)

