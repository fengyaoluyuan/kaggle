#_*_coding: utf-8_*_
# 作者     : fengyao
# 创建时间 ：2018/9/11 17:02
# 文件     ：data_to_image.py
#IDE       : PyCharm

import matplotlib.pyplot as plt
from Titanic import file_to_data


return_mat, class_labels_vector = file_to_data.file2matrix('new_train.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(return_mat[:, 2], class_labels_vector)
plt.show()