import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -----------------------------------------------------------------------------
# 1. 设置 SCI 绘图风格
# -----------------------------------------------------------------------------
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "mathtext.fontset": 'stix',
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "grid.linestyle": '--',
    "grid.alpha": 0.6
}
rcParams.update(config)

# -----------------------------------------------------------------------------
# 2. 电池模型参数定义
# -----------------------------------------------------------------------------
# 假设参数 (标称容量 Cn = 2.5 Ah, 约 9Wh 电池，对应手机/小型设备)
params = {
    'R0': 0.012,      # 欧姆内阻 (Ohm)
    'R1': 0.003,      # 极化电阻1 (Ohm)
    'C1': 4850,      # 极化电容1 (F)
    'R2': 0.002,      # 极化电阻2 (Ohm)
    'C2': 88200,      # 极化电容2 (F)
    'Cn': 3 * 3600, # 标称容量 (As, 2.5Ah)
    'SOH': 1.0,      # 健康状态
    'eta': 1.0,      # 库伦效率
    'eta_conv': 1.0, # 功率转换效率 (假设DC-DC为理想)
}

# OCV-SOC 曲线拟合函数 (根据图片3公式)
def get_ocv(soc):
    # 限制 soc 在 0-1 之间，防止多项式在物理范围外发散
    soc = np.clip(soc, 0, 1)
    return (66.939 * soc**7 - 248.739 * soc**6 + 371.911 * soc**5 
            - 291.517 * soc**4 + 132.659 * soc**3 - 36.387 * soc**2 
            + 6.069 * soc + 3.259)

# -----------------------------------------------------------------------------
# 3. 核心求解逻辑：计算电流 (解决代数环)
# -----------------------------------------------------------------------------
def solve_current(P_target, V_p1, V_p2, soc, params):
    """
    给定目标功率和当前状态，求解电流 I。
    方程: V_t = OCV - I*R0 - V_p1 - V_p2
          I = P / (eta_conv * V_t)
    代入得: P / (eta_conv * I) = OCV - I*R0 - V_p1 - V_p2
    整理为一元二次方程: R0*I^2 - (OCV - V_p1 - V_p2)*I + P/eta_conv = 0
    """
    R0 = params['R0']
    eta_conv = params['eta_conv']
    
    V_oc = get_ocv(soc)
    V_rest = V_oc - V_p1 - V_p2 # 除去欧姆降的电压
    
    # 求解二次方程 a*I^2 + b*I + c = 0
    a = R0
    b = -V_rest
    c = P_target / eta_conv
    
    delta = b**2 - 4*a*c
    
    if delta < 0:
        # 物理上意味着功率过大，电池电压崩溃，无法维持该功率
        return None, 0.0 # 电压崩溃
    
    # 取较小的根作为电流解 (较大的根对应电压极低的不稳定点)
    I = (-b - np.sqrt(delta)) / (2*a)
    
    # 计算端电压
    Vt = V_rest - I*R0
    
    return I, Vt

# -----------------------------------------------------------------------------
# 4. 状态导数计算
# -----------------------------------------------------------------------------
def state_derivatives(t, state, P_target, params):
    """
    计算状态变量的导数: [dSOC/dt, dVp1/dt, dVp2/dt]
    state = [SOC, Vp1, Vp2]
    """
    SOC, Vp1, Vp2 = state
    
    if SOC <= 0:
        return np.array([0.0, 0.0, 0.0]) # 电池耗尽，停止变化

    # 1. 求解代数环得到电流 I
    I, Vt = solve_current(P_target, Vp1, Vp2, SOC, params)
    
    if I is None or Vt < 2.5: # 截止电压保护
        return np.array([0.0, 0.0, 0.0]) # 停止放电

    # 2. 计算导数 (图片1公式)
    # dSOC/dt = -eta * I / (Cn * SOH)
    dSOC_dt = -params['eta'] * I / (params['Cn'] * params['SOH'])
    
    # dVp1/dt = I/C1 - Vp1/(R1*C1)
    dVp1_dt = I / params['C1'] - Vp1 / (params['R1'] * params['C1'])
    
    # dVp2/dt = I/C2 - Vp2/(R2*C2)
    dVp2_dt = I / params['C2'] - Vp2 / (params['R2'] * params['C2'])
    
    return np.array([dSOC_dt, dVp1_dt, dVp2_dt])

# -----------------------------------------------------------------------------
# 5. 数值积分方法实现
# -----------------------------------------------------------------------------
def run_simulation(method, power, dt=1.0):
    # 初始状态
    t_curr = 0
    soc_curr = 1.0 # 初始 SOC 100%
    vp1_curr = 0.0
    vp2_curr = 0.0
    
    time_list = [0]
    soc_list = [1.0]
    
    max_time = 20 * 3600 # 最大仿真时间 (秒)
    
    while soc_curr > 0 and t_curr < max_time:
        state_curr = np.array([soc_curr, vp1_curr, vp2_curr])
        
        # 检查是否电压截止
        I_check, Vt_check = solve_current(power, vp1_curr, vp2_curr, soc_curr, params)
        if I_check is None or Vt_check < 2.7: # 截止电压 2.7V
            break

        if method == 'Euler':
            # --- 前向欧拉法 ---
            grads = state_derivatives(t_curr, state_curr, power, params)
            state_next = state_curr + grads * dt
            
        elif method == 'RK4':
            # --- RK4 法 ---
            k1 = state_derivatives(t_curr, state_curr, power, params)
            k2 = state_derivatives(t_curr + dt/2, state_curr + k1 * dt/2, power, params)
            k3 = state_derivatives(t_curr + dt/2, state_curr + k2 * dt/2, power, params)
            k4 = state_derivatives(t_curr + dt, state_curr + k3 * dt, power, params)
            
            state_next = state_curr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 更新状态
        soc_curr, vp1_curr, vp2_curr = state_next
        t_curr += dt
        
        time_list.append(t_curr / 3600.0) # 转为小时
        soc_list.append(max(0, soc_curr)) # 保证SOC不小于0
        
    return time_list, soc_list

# -----------------------------------------------------------------------------
# 6. 主程序：定义场景并绘图
# -----------------------------------------------------------------------------
scenarios = [
    {"name": "Standby", "power": 0.55, "color": "#1f77b4", "linestyle": "-"},
    {"name": "Web Browsing", "power": 1.35, "color": "#2ca02c", "linestyle": "-"},
    {"name": "Video Streaming (4K)", "power": 4.66, "color": "#ff7f0e", "linestyle": "-"},
    {"name": "Heavy Gaming", "power": 6.05, "color": "#d62728", "linestyle": "-"},
    {"name": "Video Recording (4K)", "power": 7.46, "color": "#9467bd", "linestyle": "-"}
]

# 创建图形窗口 (两个子图：左边欧拉法，右边RK4法)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# --- 绘制图 1：Euler Method ---
for s in scenarios:
    t_data, soc_data = run_simulation('Euler', s['power'], dt=1.0) # dt=1s
    ax1.plot(t_data, soc_data, label=f"{s['name']} ({s['power']}W)", 
             color=s['color'], linewidth=1.8, linestyle=s['linestyle'])

ax1.set_xlabel("Time (h)")
ax1.set_ylabel("State of Charge (SOC)")
ax1.set_title("(a) SOC-Time Curves (Euler Method)", fontweight='bold', pad=12)
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 1.05)
ax1.grid(True)
ax1.legend(loc='lower left', frameon=True, edgecolor='black', fancybox=False)

# --- 绘制图 2：RK4 Method ---
for s in scenarios:
    t_data, soc_data = run_simulation('RK4', s['power'], dt=5.0) # RK4 步长可以稍大
    ax2.plot(t_data, soc_data, label=f"{s['name']} ({s['power']}W)", 
             color=s['color'], linewidth=1.8, linestyle=s['linestyle'])

ax2.set_xlabel("Time (h)")
ax2.set_ylabel("State of Charge (SOC)")
ax2.set_title("(b) SOC-Time Curves (RK4 Method)", fontweight='bold', pad=12)
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 1.05)
ax2.grid(True)
ax2.legend(loc='lower left', frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig('SOC_Time_Comparison_SCI_v2.png', bbox_inches='tight')
plt.show()

print("仿真完成。图表已生成。")
