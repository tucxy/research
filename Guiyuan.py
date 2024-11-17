from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# 中点和斜率参数
u1 = 1
u2 = 4
alpha = np.pi / 6
sigma = 2
# 计算类中心
positive_center_1 = (u1 + u2 * np.cos(alpha), u2 * np.sin(alpha))
positive_center_2 = (u1 - u2 * np.cos(alpha), -u2 * np.sin(alpha))
negative_center_1 = (-u1 + u2 * np.cos(alpha), u2 * np.sin(alpha))
negative_center_2 = (-u1 - u2 * np.cos(alpha), -u2 * np.sin(alpha))

# 协方差矩阵

cov_matrix = [[sigma**2, 0], [0, sigma**2]]
rv_positive1 = multivariate_normal(positive_center_1, cov_matrix)
rv_positive2 = multivariate_normal(positive_center_2, cov_matrix)
rv_negative1 = multivariate_normal(negative_center_1, cov_matrix)
rv_negative2 = multivariate_normal(negative_center_2, cov_matrix)

# 正类和负类的混合密度函数

def f_positive(x, y):
    return 0.5 * rv_positive1.pdf([x, y]) + 0.5 *
rv_positive2.pdf([x, y])
def f_negative(x, y):
    return 0.5 * rv_negative1.pdf([x, y]) + 0.5 *
rv_negative2.pdf([x, y])

# 计算三个线段
mid_left = ((positive_center_2[0] + negative_center_2[0]) / 2,
(positive_center_2[1] + negative_center_2[1]) / 2)
mid_right = ((positive_center_1[0] + negative_center_1[0]) / 2,
(positive_center_1[1] + negative_center_1[1]) / 2)
mid_center = ((positive_center_2[0] + negative_center_1[0]) / 2,
(positive_center_2[1] + negative_center_1[1]) / 2)
# 中线的斜率
center_slope = (positive_center_2[1] - negative_center_1[1]) /
(positive_center_2[0] - negative_center_1[0])
perpendicular_slope = -1 / center_slope # 垂直线的斜率
# 定义边界曲线
def boundary_curve(x):

    return (mid_right[1] / mid_right[0]) * x
# 定义积分函数,y 从边界曲线延伸到 y=-10
def integrand_pos_region(x, y):
    return f_positive(x, y)
# 计算第一个区域的体积
est_pos_volume_region, error_pos_region = dblquad(
integrand_pos_region,
0, mid_right[0], # x 从 0 到 mid_right[0]
lambda x: -10, # y 从 -10
boundary_curve # y 的上界是边界曲线
)
# print("Estimated Positive Volume for Region (First Part):",est_pos_volume_region)
# print("Estimation Error for Region (First Part):",error_pos_region)
# 第二部分代码:计算右侧的实际边界的体积
# 向量化密度函数以支持网格输入
f_positive_vec = np.vectorize(f_positive)
f_negative_vec = np.vectorize(f_negative)
# 计算网格上的 f_positive / f_negative 值
x_vals = np.linspace(-10, 10, 500)
y_vals = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x_vals, y_vals)
ratio = f_positive_vec(X, Y) / f_negative_vec(X, Y)
# 绘制等高线并找到真实边界
plt.figure(figsize=(10, 6))
contour_boundary = plt.contour(X, Y, ratio, levels=[1],
colors='green', linewidths=1)
plt.clabel(contour_boundary, inline=True, fontsize=10) = 1
# 提取等高线数据,移除重复 x 值
paths = contour_boundary.collections[0].get_paths()
boundary_coords = paths[0].vertices
x_boundary, y_boundary = boundary_coords[:, 0], boundary_coords[:,
1]
# 移除重复的 x 值
unique_x_boundary, unique_indices = np.unique(x_boundary,
return_index=True)
x_boundary = unique_x_boundary
y_boundary = y_boundary[unique_indices]
# 插值生成平滑的边界函数

real_boundary = interp1d(x_boundary, y_boundary, kind='cubic',
fill_value="extrapolate")
# 找到真实边界曲线的右端点 x 值(对应 y = 10 时的 x)
x_boundary_max = x_boundary.max()
def integrand_pos_right_region(y, x):
    return f_positive(x, y)
# 真实体积计算的积分区域 est_real_volume_upper, error_real_upper = dblquad(integrand_pos_right_region,0, x_boundary_max, # x 从0到 x_boundary_maxlambda x: -10, # y 的下界为 -10 lambda x: real_boundary(x) # y 的上界为真实边界函数real_boundary(x))
# 输出结果
# print("Estimated Positive Volume on the Right Side of Real Boundary:", est_real_volume_upper)
# print("Estimation Error on the Right Side of Real Boundary:",error_real_upper)

half_misclassification = est_real_volume_upper-est_pos_volume_region

print("half misclassification:", half_misclassification)
# full_misclassification = 2*half_misclassification
# print("full misclassification:", full_misclassification)
volume_ratio = est_pos_volume_region / est_real_volume_upper
print("Volume ratio ( First Region Volume / Right Boundary Volume):", volume_ratio)