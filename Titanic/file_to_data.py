#_*_coding: utf-8_*_
# 作者     : fengyao
# 创建时间 ：2018/9/11 10:24
# 文件     ：file_to_data.py
#IDE       : PyCharm

import numpy as np


N = 8

def train_file2matrix(filename):
    '''
    从文件中获取训练集数据
    :param filename:
    :return:
    '''
    train = np.loadtxt(filename, delimiter=',')
    return_mat = train[:, : N].reshape((len(train), N))
    class_labels_vector = train[:, -1:].reshape((len(train), 1))
    return return_mat, class_labels_vector

def test_file2matrix(filename):
    test = np.loadtxt(filename, delimiter=',')
    return_mat = test[:, : N].reshape(len(test), N)
    class_labels_vector = test[:, -1:].reshape((len(test), 1))
    return return_mat, class_labels_vector

def main():
    train_filename = 'train.csv'
    train_data, train_labels = train_file2matrix(filename=train_filename)
    print(train_data)
    print(train_labels)
    test_filename = 'test.csv'
    test_data, test_labels = test_file2matrix(filename=test_filename)
    print(test_data)
    print(test_labels)

if __name__ == '__main__':
    main()