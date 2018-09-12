#_*_coding: utf-8_*_
# 作者     : fengyao
# 创建时间 ：2018/9/11 23:56
# 文件     ：test.py
#IDE       : PyCharm

import numpy as np
from Titanic import file_to_data
from Titanic import auto_normal
from Titanic import class_type
import operator


# def test():
#     train_data, train_labels = file_to_data.train_file2matrix('train.csv')
#     test_data, test_labels = file_to_data.test_file2matrix('test.csv')
#     train_norm_data, train_ranges, train_min_values = auto_normal.auto_norm(train_data)
#     test_norm_data, test_ranges, test_min_values = auto_normal.auto_norm(test_data)
#     m_test = test_norm_data.shape[0]
#     error_count = 0.0
#     test_result = []
#     for i in range(m_test):
#         example = test_norm_data[i]
#         example_result = int(class_type.classfy1(train_norm_data, train_labels, example, 100))
#         print('预测结果为：%d, 实际结果为：%d' %(example_result, test_labels[i]))
#         test_result.append(example_result)
#         if example_result != test_labels[i]:
#             error_count += 1
#     error_rate = error_count / float(m_test)
#     np.savetxt('test_result.csv', test_result, delimiter=',')
#     print('错误率：', error_rate)

def test():
    train_data, train_labels = file_to_data.train_file2matrix('train.csv')
    test_data, test_labels = file_to_data.test_file2matrix('test.csv')
    train_norm_data, train_ranges, train_min_values = auto_normal.auto_norm(train_data)
    test_norm_data, test_ranges, test_min_values = auto_normal.auto_norm(test_data)
    m_test = test_norm_data.shape[0]
    error_count = 0.0
    test_result = []
    error_rate_dict = {}
    for k in range(5, int(m_test / 2)):
        for i in range(m_test):
            example = test_norm_data[i]
            example_result = int(class_type.classfy1(train_norm_data, train_labels, example, k))
            #print('预测结果为：%d, 实际结果为：%d' % (example_result, test_labels[i]))
            test_result.append(example_result)
            if example_result != test_labels[i]:
                error_count += 1
        error_rate = error_count / float(m_test)
        #print('错误率：', error_rate)
        # np.savetxt('test_result.csv', test_result, delimiter=',')
        error_rate_dict[k] = error_rate
        #print(error_rate_dict)
    sort_error_rate = sorted(error_rate_dict.items(), key=lambda x: x[1])
    print(sort_error_rate[0][1])

# def main():
#     k = 100
#     test(k)
#
# if __name__ == '__main__':
#     main()