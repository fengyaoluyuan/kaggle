#_*_coding: utf-8_*_
# 作者     : fengyao
# 创建时间 ：2018/9/11 17:21
# 文件     ：auto_normal.py
#IDE       : PyCharm

import numpy as np
from Titanic import file_to_data


def auto_norm(return_mat):
    min_values = return_mat.min(0)
    max_values = return_mat.max(0)
    ranges = max_values - min_values
    m = return_mat.shape[0]
    norm_data = return_mat - np.tile(min_values, (m, 1))
    norm_data = norm_data / np.tile(ranges, (m, 1))
    return norm_data, ranges, min_values

def main():
    train_data, train_labels = file_to_data.train_file2matrix('train.csv')
    test_data, test_labels = file_to_data.test_file2matrix('test.csv')
    train_norm_data, train_ranges, train_min_values = auto_norm(train_data)
    test_norm_data, test_ranges, test_min_values = auto_norm(test_data)
    print(train_norm_data, train_ranges, train_min_values)
    print(test_norm_data, test_ranges, test_min_values)

if __name__ == '__main__':
    main()