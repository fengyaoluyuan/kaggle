#_*_coding: utf-8_*_
# 作者     : fengyao
# 创建时间 ：2018/9/11 17:38
# 文件     ：class_type.py
#IDE       : PyCharm

import numpy as np
import operator
from Titanic import file_to_data
from Titanic import auto_normal


def classfy1(train_data, labels, example, k):
    data_train_row = train_data.shape[0]
    diff_mat = np.tile(example, (data_train_row, 1)) - train_data
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distance = sq_distance ** 0.5
    distance_labels = np.column_stack((distance, labels))
    sort_distance_increase = distance_labels[np.lexsort(distance_labels.T[0, None])]
    result = sort_distance_increase[:, 1:]
    class_count = {}
    for i in range(k):
        vote_label = result[i][0]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    print(sort_class_count)
    return sort_class_count[0][0]

def main():
    train_data, labels = file_to_data.train_file2matrix('train.csv')
    test_data = file_to_data.test_file2matrix('test.csv')
    train_norm_data, train_ranges, train_min_values = auto_normal.auto_norm(train_data)
    test_norm_data, test_ranges, test_min_values = auto_normal.auto_norm(test_data)
    example = test_norm_data[0]
    k = 10
    result = classfy1(train_norm_data, labels, example, k)
    print(result)

if __name__ == '__main__':
    main()
