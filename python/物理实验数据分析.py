import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

#原始数据
x_data = np.array([
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
    20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
    30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
    40.0, 41.0
])
I_ratio_data = np.array([
    0.000, 0.000, 0.000, 0.006, 0.005, 0.000, 0.000, 0.000, 0.013, 0.025,
    0.025, 0.009, 0.000, 0.014, 0.087, 0.229, 0.412, 0.579, 1.000, 0.626,
    0.45, 0.267, 0.125, 0.028, 0.000, 0.006, 0.02, 0.024, 0.012, 0.000,
    0.000, 0.000, 0.004, 0.005, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000
])
a = 0.106e-3 
I0 = 11352.1   
wavelength = 632.8e-9  
L = 1.0       
x_theory = np.linspace(0, 41, 1000) 
x_prime = (x_theory - 18.0) / 1000  
beta = (np.pi * a * x_prime) / (wavelength * L)
I_ratio_theory = (np.sin(beta) / beta)**2
I_ratio_theory[np.isnan(I_ratio_theory)] = 1.0 

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(x_data, I_ratio_data, color='red', label='实验数据', zorder=2)
plt.plot(x_theory, I_ratio_theory, color='blue', label=f'理论曲线 (a={a*1e3:.3f}mm)', linewidth=1.5)
plt.xlabel('位置 x (mm)', fontsize=12)
plt.ylabel('归一化光强 I/I_0', fontsize=12)
plt.title('单缝衍射光强分布 (I/I_0 - x 曲线)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.axvline(x=18.0, color='gray', linestyle='--', label='中央极大 (x=18.0mm)')
plt.axvline(x=12.0, color='green', linestyle=':', label='第一极小 (x=12.0/24.0mm)')
plt.axvline(x=24.0, color='green', linestyle=':')
plt.xlim(0, 41)
plt.ylim(0, 1.1)
plt.show()