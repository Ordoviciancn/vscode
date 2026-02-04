import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
import os

# 创建保存图片的文件夹
output_dir = "battery_figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. 修正后的模型参数 ---
# 假设电池：3000mAh, 标称3.7V, 能量约11.1Wh
Q_capacity_Ah = 3.0 
Q_capacity_C = Q_capacity_Ah * 3600 # 库伦

# 更鲁棒的OCV曲线 (基于经验公式，而非可能过拟合的多项式)
def get_ocv(soc):
    # 限制 soc 在 0-1 之间，防止多项式在物理范围外发散
    soc = np.clip(soc, 0, 1)
    return (66.939 * soc**7 - 248.739 * soc**6 + 371.911 * soc**5 
            - 291.517 * soc**4 + 132.659 * soc**3 - 36.387 * soc**2 
            + 6.069 * soc + 3.259)
# 物理参数
R0 = 0.012   # 欧姆 (内阻)
R1 = 0.003; C1 = 4850
R2 = 0.002; C2 = 88200
V_cutoff = 3.0 # 截止电压

# --- 2. 核心动力学 ---
def battery_dynamics(t, y, P_load):
    soc, vp1, vp2 = y
    
    if soc <= 0: return [0, 0, 0] # 保护
    
    ocv = get_ocv(soc)
    
    # 解代数环: V_term^2 - (OCV-Vp1-Vp2)*V_term + P*R0 = 0
    U_effective = ocv - vp1 - vp2
    delta = U_effective**2 - 4 * P_load * R0
    
    if delta < 0:
        # 功率过大，电压崩溃
        i_batt = P_load / V_cutoff # 近似处理
    else:
        v_term = (U_effective + np.sqrt(delta)) / 2
        i_batt = P_load / v_term
    
    d_soc = -i_batt / Q_capacity_C
    d_vp1 = (i_batt/C1) - (vp1/(R1*C1))
    d_vp2 = (i_batt/C2) - (vp2/(R2*C2))
    
    return [d_soc, d_vp1, d_vp2]

# --- 3. 改进的求解器 ---
def simulate_discharge(P_load, soc_init=1.0):
    t_span = [0, 50 * 3600] # 最多跑50小时
    y0 = [soc_init, 0, 0]
    
    # 终止事件：电压低于截止电压
    def voltage_cutoff(t, y):
        soc, vp1, vp2 = y
        U_eff = get_ocv(soc) - vp1 - vp2
        delta = U_eff**2 - 4 * P_load * R0
        if delta < 0: return -1.0
        v_term = (U_eff + np.sqrt(delta)) / 2
        return v_term - V_cutoff
    
    voltage_cutoff.terminal = True
    voltage_cutoff.direction = -1
    
    # 终止事件：SOC低于0
    def soc_cutoff(t, y):
        return y[0]
    soc_cutoff.terminal = True
    
    sol = solve_ivp(
        lambda t,y: battery_dynamics(t,y,P_load),
        t_span, y0, events=[voltage_cutoff, soc_cutoff],
        method='RK45', rtol=1e-4, atol=1e-6, max_step=60
    )
    
    # 重建电压曲线用于绘图
    time_h = sol.t / 3600
    soc_traj = sol.y[0]
    v_traj = []
    for i, s in enumerate(soc_traj):
        U_eff = get_ocv(s) - sol.y[1][i] - sol.y[2][i]
        delta = U_eff**2 - 4 * P_load * R0
        if delta < 0: v_traj.append(V_cutoff)
        else: v_traj.append((U_eff + np.sqrt(delta)) / 2)
        
    tte = time_h[-1] if sol.status == 1 else time_h[-1] # 如果没触发事件，取最后时间
    
    return time_h, soc_traj, np.array(v_traj), tte

# --- 绘图设置 (SCI标准) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 10

# ================================
# 图1: 不同功率下的放电曲线 (Discharge Profiles)
# ================================
fig1 = plt.figure(figsize=(7, 5))
ax1 = fig1.add_subplot(1, 1, 1)
powers = [0.5, 1.5, 3.0, 10.0]
colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

for p, c in zip(powers, colors):
    t, s, v, tte = simulate_discharge(p)
    ax1.plot(t, v, label=f'P={p}W (TTE={tte:.1f}h)', color=c, linewidth=2)

ax1.axhline(V_cutoff, color='k', linestyle='--', linewidth=1, label='Cutoff Voltage')
ax1.set_xlabel('Time (Hours)', fontweight='bold')
ax1.set_ylabel('Terminal Voltage (V)', fontweight='bold')
ax1.set_title('Voltage Discharge Profiles under Various Power Loads', fontweight='bold')
ax1.legend(frameon=False, fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xlim(left=0)
plt.tight_layout()
plt.savefig(f'{output_dir}/Figure_a_Voltage_Profiles.png', dpi=300)
plt.close(fig1)

# ================================
# 图2: SOC与电压的双轴关系 (SOC vs Voltage)
# ================================
fig2 = plt.figure(figsize=(7, 5))
ax2 = fig2.add_subplot(1, 1, 1)
t, s, v, tte = simulate_discharge(3.0) # 3W
ax2.plot(t, s*100, color='#8e44ad', linewidth=2, label='SOC (%)')
ax2.set_ylabel('State of Charge (%)', fontweight='bold', color='#8e44ad')
ax2.tick_params(axis='y', labelcolor='#8e44ad')
ax2.set_xlabel('Time (Hours)', fontweight='bold')

ax2_r = ax2.twinx()
ax2_r.plot(t, v, color='#2c3e50', linewidth=2, linestyle='-', label='Voltage (V)')
ax2_r.set_ylabel('Voltage (V)', fontweight='bold', color='#2c3e50')
ax2_r.tick_params(axis='y', labelcolor='#2c3e50')

# 添加图例
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)

ax2.set_title('Dual-Axis: SOC & Voltage Dynamics (3W Load)', fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(f'{output_dir}/Figure_b_Dual_Axis.png', dpi=300)
plt.close(fig2)

# ================================
# 图3: TTE Landscape (修正后的热力图，横轴范围缩短到4W)
# ================================
fig3 = plt.figure(figsize=(7, 5))
ax3 = fig3.add_subplot(1, 1, 1)

# 修改横轴范围为0.2-4.0W（原为0.2-6.0W）
soc_vals = np.linspace(0.1, 1.0, 15)
p_vals = np.linspace(0.2, 3.0, 15)  # 修改这里：最大4W
X, Y = np.meshgrid(p_vals, soc_vals)
Z = np.zeros_like(X)

# 计算TTE
for i in range(len(soc_vals)):
    for j in range(len(p_vals)):
        _, _, _, tte = simulate_discharge(p_vals[j], soc_init=soc_vals[i])
        Z[i, j] = tte

# 创建热力图
cp = ax3.contourf(X, Y, Z, levels=50, cmap='viridis')  # levels从15→50，热力分层更细
cbar = fig3.colorbar(cp, ax=ax3)
cbar.set_label('TTE (Hours)', rotation=270, labelpad=15)

# 等高线levels从8→30，白色线条更密集，保留原有格式
CS = ax3.contour(X, Y, Z, levels=15, colors='w', linewidths=0.5, alpha=0.8)
ax3.clabel(CS, inline=True, fontsize=5, fmt='%.1f h')

ax3.set_xlabel('Constant Power Load (W)', fontweight='bold')
ax3.set_ylabel('Initial SOC', fontweight='bold')
ax3.set_title('TTE Prediction Landscape', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/Figure_c_TTE_Landscape.png', dpi=300)
plt.close(fig3)

# ================================
# 图4: 蒙特卡洛不确定性 (修正后的分布)
# ================================
fig4 = plt.figure(figsize=(7, 5))
ax4 = fig4.add_subplot(1, 1, 1)
tte_dist = []
for _ in range(500):
    # 模拟功率波动: 均值3W, 标准差0.5W
    p_rand = np.random.normal(3.0, 0.5)
    if p_rand < 0.1: p_rand = 0.1
    _, _, _, tte = simulate_discharge(p_rand, soc_init=0.8)
    tte_dist.append(tte)

sns.histplot(tte_dist, kde=True, color='#34495e', element="step", ax=ax4)
mu = np.mean(tte_dist)
ci_low = np.percentile(tte_dist, 2.5)
ci_high = np.percentile(tte_dist, 97.5)

ax4.axvline(mu, color='r', linestyle='--', label=f'Mean: {mu:.2f}h')
ax4.axvspan(ci_low, ci_high, color='orange', alpha=0.2, label='95% CI')
ax4.set_xlabel('Time-to-Empty (Hours)', fontweight='bold')
ax4.set_title('Prediction Uncertainty under Fluctuating Load (Initial SOC=0.8)', fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{output_dir}/Figure_d_Uncertainty.png', dpi=300)
plt.close(fig4)

print(f"所有图表已保存到 '{output_dir}' 文件夹中:")
print(f"1. {output_dir}/Figure_a_Voltage_Profiles.png")
print(f"2. {output_dir}/Figure_b_Dual_Axis.png")
print(f"3. {output_dir}/Figure_c_TTE_Landscape.png")
print(f"4. {output_dir}/Figure_d_Uncertainty.png")