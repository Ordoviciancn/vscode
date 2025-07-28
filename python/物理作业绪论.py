import numpy as np

d_5, d_10, d_15, d_20, d_25, d_30 = 23.532, 22.840, 22.375, 21.912, 21.556, 21.226
d_5_prime, d_10_prime, d_15_prime, d_20_prime, d_25_prime, d_30_prime = 26.679, 27.366, 27.865, 28.316, 28.674, 29.030

m_values = np.array([5, 10, 15, 20, 25, 30])
r_values = np.abs(np.array([d_5, d_10, d_15, d_20, d_25, d_30]) - np.array([d_5_prime, d_10_prime, d_15_prime, d_20_prime, d_25_prime, d_30_prime])) / 2

r_squared_values = r_values ** 2

x_mean = np.mean(m_values)
y_mean = np.mean(r_squared_values)

numerator = np.sum((m_values - x_mean) * (r_squared_values - y_mean))
denominator = np.sum((m_values - x_mean) ** 2)
k = numerator / denominator

lambda_value = 589.3 * 10 ** (-6)
R = k / lambda_value

y_fit = k * m_values
S_y = np.sqrt(np.sum((r_squared_values - y_fit) ** 2) / (len(m_values) - 2))

u_k = S_y / np.sqrt(denominator)

u_R = u_k / lambda_value

print("平凸透镜的曲率半径 R:", R, "mm")
print("曲率半径 R 的不确定度:", u_R, "mm")