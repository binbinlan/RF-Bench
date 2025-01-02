import numpy as np
import matplotlib.pyplot as plt

# 创建一个二维矩阵
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

row_sums = np.sum(matrix, axis=1)
max_row_index = np.argmax(row_sums)
print('最大索引是',max_row_index)
row_sums_list = row_sums.tolist()

print(row_sums_list)
# 创建热图
plt.imshow(matrix, cmap='viridis', interpolation='nearest')

# 添加颜色条
plt.colorbar()

# 添加标题和标签
plt.title('Heatmap of 2D Matrix')
plt.xlabel('Columns')
plt.ylabel('Rows')

# 设置 x 和 y 轴的刻度
plt.xticks(ticks=np.arange(matrix.shape[1]), labels=np.arange(1, matrix.shape[1] + 1))
plt.yticks(ticks=np.arange(matrix.shape[0]), labels=np.arange(1, matrix.shape[0] + 1))

# 显示图形
plt.show()