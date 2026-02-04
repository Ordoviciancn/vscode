# 第一步：解决乱码/警告
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# 解决TensorFlow日志/中文/编码问题
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['image.interpolation'] = 'nearest'
os.environ['MPLCONFIGDIR'] = os.getcwd()
sys.stdout.reconfigure(encoding='utf-8')

# 第二步：准备数据
data = {
    'year': [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
    'medals': [81, 174, 94, 108, 101, 97, 101, 110, 104, 121, 113, 106]
}
df = pd.DataFrame(data)
time_series = df['medals'].values.reshape(-1, 1)

# 第三步：标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series)

# 第四步：构建输入数据
n_steps = 5
X, y = [], []
for i in range(n_steps, len(scaled_data)):
    X.append(scaled_data[i-n_steps:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 第五步：拆分训练/测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 第六步：构建LSTM模型（修复Input层警告）
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 第七步：训练模型
print("开始训练LSTM模型...")
history = model.fit(
    X_train, y_train,
    batch_size=2,
    epochs=100,
    validation_data=(X_test, y_test)
)

# 第八步：预测
# 测试集预测
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# 2028年预测
last_5_steps = scaled_data[-n_steps:]
last_5_steps = np.reshape(last_5_steps, (1, n_steps, 1))
medal_2028_scaled = model.predict(last_5_steps)
medal_2028 = scaler.inverse_transform(medal_2028_scaled)
print(f"2028年美国奥运总奖牌数预测值：{medal_2028[0][0]:.2f}")

# 第九步：可视化
plt.figure(figsize=(12, 6))
# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型训练损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()

# 真实值vs预测值
plt.subplot(1, 2, 2)
plt.plot(y_test_original, label='真实奖牌数')
plt.plot(test_predict, label='预测奖牌数')
plt.axhline(y=medal_2028[0][0], color='r', linestyle='--', label='2028预测值')
plt.title('测试集：真实值 vs 预测值')
plt.xlabel('测试样本')
plt.ylabel('奖牌数')
plt.legend()

plt.tight_layout()
plt.show()