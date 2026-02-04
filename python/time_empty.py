import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.integrate import solve_ivp
import matplotlib.patches as mpatches
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== 创建输出文件夹 ====================
OUTPUT_DIR = Path("TTE_Simulation_Results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 设置SCI绘图风格 ====================
def setup_plot_style():
    """设置符合SCI期刊要求的绘图风格"""
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        
        # 图形尺寸 (适合单栏/双栏)
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        
        # 线条样式
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        
        # 坐标轴
        'axes.linewidth': 1.2,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.spines.top': True,
        'axes.spines.right': True,
        
        # 刻度
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # 网格
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        
        # 图例
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',
        
        # 保存设置
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

setup_plot_style()

# ==================== 定义SCI期刊配色方案 ====================
class SCIColors:
    """Nature/Science风格配色"""
    # 主色调
    blue = '#0072B2'
    orange = '#D55E00'
    green = '#009E73'
    red = '#CC79A7'
    purple = '#9467BD'
    yellow = '#F0E442'
    cyan = '#56B4E9'
    gray = '#999999'
    
    # 渐变色
    blues = ['#C6DBEF', '#6BAED6', '#2171B5', '#08306B']
    reds = ['#FCBBA1', '#FB6A4A', '#CB181D', '#67000D']
    greens = ['#C7E9C0', '#74C476', '#238B45', '#00441B']
    
    @classmethod
    def get_palette(cls, n=5):
        """获取n个颜色的调色板"""
        colors = [cls.blue, cls.orange, cls.green, cls.red, cls.purple, cls.cyan]
        return colors[:n]

# ==================== 电池参数类 ====================
class BatteryParameters:
    """电池模型基础参数"""
    def __init__(self):
        # 电池额定参数
        self.nominal_capacity = 3.0  # Ah
        self.nominal_voltage = 3.7   # V
        self.cutoff_voltage = 3.0    # V
        
        # 二阶RC模型参数
        self.R0 = 0.012      # 欧姆内阻 (Ω)
        self.R1 = 0.003     # 极化电阻1 (Ω)
        self.C1 = 4850      # 极化电容1 (F), τ1 ≈ 30s
        self.R2 = 0.002     # 极化电阻2 (Ω)
        self.C2 = 88200     # 极化电容2 (F), τ2 ≈ 300s
        
        # 7阶OCV-SOC多项式系数 (从高次到低次)
        self.ocv_coeffs = np.array([
            66.939, -248.739, 371.911, -291.517, 
            132.659, -36.387, 6.069, 3.259
        ])
        
        # 初始状态
        self.soc_initial = 1.0
        self.v1_initial = 0.0
        self.v2_initial = 0.0
        
        # 派生参数
        self.q_max = self.nominal_capacity * 3600  # 库仑
        
    def ocv_from_soc(self, soc):
        """从SOC计算开路电压"""
        soc_clipped = np.clip(soc, 0.0, 1.0)
        return np.polyval(self.ocv_coeffs, soc_clipped)
    
    def copy(self):
        """创建参数副本"""
        new_params = BatteryParameters()
        new_params.R0 = self.R0
        new_params.R1 = self.R1
        new_params.C1 = self.C1
        new_params.R2 = self.R2
        new_params.C2 = self.C2
        new_params.nominal_capacity = self.nominal_capacity
        new_params.q_max = self.nominal_capacity * 3600
        return new_params

# ==================== 二阶RC模型（优化版）====================
class SecondOrderRCModel:
    """二阶RC等效电路模型 - 优化版"""
    
    def __init__(self, params):
        self.params = params
        
    def solve_current(self, soc, v1, v2, P):
        """
        求解给定功率下的电流
        代数方程: P = (OCV - I*R0 - V1 - V2) * I
        """
        ocv = self.params.ocv_from_soc(soc)
        V_oc_eff = ocv - v1 - v2  # 有效开路电压
        
        # 二次方程: R0*I² - V_oc_eff*I + P = 0
        a = self.params.R0
        b = -V_oc_eff
        c = P
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            # 功率不可达，返回最大可能电流
            I_max = V_oc_eff / (2 * self.params.R0)
            return max(0, I_max)
        
        sqrt_disc = np.sqrt(discriminant)
        I1 = (-b + sqrt_disc) / (2*a)
        I2 = (-b - sqrt_disc) / (2*a)
        
        # 选择物理上合理的解（正电流，且产生正端电压）
        candidates = []
        for I in [I1, I2]:
            if I >= 0:
                V_term = V_oc_eff - I * self.params.R0
                if V_term > 0:
                    candidates.append((I, V_term))
        
        if not candidates:
            return 0.0
        
        # 返回电流较小的解（效率更高）
        return min(candidates, key=lambda x: x[0])[0]
    
    def model_ode(self, t, state, power_func):
        """ODE右端函数"""
        soc, v1, v2 = state
        
        # 限制状态范围
        soc = np.clip(soc, 0.001, 1.0)
        
        # 获取功率
        P = power_func(t) if callable(power_func) else power_func
        
        # 求解电流
        I = self.solve_current(soc, v1, v2, P)
        
        # 状态微分
        dsoc_dt = -I / self.params.q_max
        dv1_dt = -v1 / (self.params.R1 * self.params.C1) + I / self.params.C1
        dv2_dt = -v2 / (self.params.R2 * self.params.C2) + I / self.params.C2
        
        return [dsoc_dt, dv1_dt, dv2_dt]
    
    def compute_voltage(self, soc, v1, v2, I):
        """计算端电压"""
        ocv = self.params.ocv_from_soc(soc)
        return ocv - I * self.params.R0 - v1 - v2
    
    def simulate(self, power, t_max=None, dt=1.0, verbose=False):
        """
        简化且稳定的仿真方法
        
        Args:
            power: 功率值(W)或功率函数
            t_max: 最大仿真时间(s)，默认根据功率估算
            dt: 时间步长(s)
            verbose: 是否打印详细信息
            
        Returns:
            tte: 放电时间(h)
            results: 结果字典
        """
        # 估算最大仿真时间
        if t_max is None:
            avg_power = power if not callable(power) else 3.0
            energy = self.params.nominal_capacity * self.params.nominal_voltage
            t_max = 2 * energy / avg_power * 3600  # 2倍估算时间
        
        # 初始状态
        state = np.array([
            self.params.soc_initial,
            self.params.v1_initial,
            self.params.v2_initial
        ])
        
        # 结果存储
        times = [0.0]
        socs = [state[0]]
        v1s = [state[1]]
        v2s = [state[2]]
        
        # 计算初始电流和电压
        P0 = power(0) if callable(power) else power
        I0 = self.solve_current(state[0], state[1], state[2], P0)
        V0 = self.compute_voltage(state[0], state[1], state[2], I0)
        
        currents = [I0]
        voltages = [V0]
        
        # 欧拉法时间积分（简单稳定）
        t = 0.0
        tte = t_max / 3600
        
        while t < t_max:
            # 获取当前功率
            P = power(t) if callable(power) else power
            
            # 求解电流
            I = self.solve_current(state[0], state[1], state[2], P)
            
            # 计算端电压
            V_term = self.compute_voltage(state[0], state[1], state[2], I)
            
            # 检查截止条件
            if V_term <= self.params.cutoff_voltage or state[0] <= 0.001:
                tte = t / 3600
                break
            
            # 状态更新（欧拉法）
            dsoc_dt = -I / self.params.q_max
            dv1_dt = -state[1] / (self.params.R1 * self.params.C1) + I / self.params.C1
            dv2_dt = -state[2] / (self.params.R2 * self.params.C2) + I / self.params.C2
            
            state[0] += dsoc_dt * dt
            state[1] += dv1_dt * dt
            state[2] += dv2_dt * dt
            
            # 限制状态范围
            state[0] = np.clip(state[0], 0.0, 1.0)
            
            t += dt
            
            # 存储结果（降采样）
            if len(times) < 5000 or t >= times[-1] + 10:  # 最多5000个点或每10秒存一次
                times.append(t)
                socs.append(state[0])
                v1s.append(state[1])
                v2s.append(state[2])
                currents.append(I)
                voltages.append(V_term)
        
        results = {
            'time': np.array(times),
            'soc': np.array(socs),
            'v1': np.array(v1s),
            'v2': np.array(v2s),
            'current': np.array(currents),
            'voltage': np.array(voltages),
            'tte': tte
        }
        
        if verbose:
            print(f"TTE: {tte:.3f} h, Final SOC: {socs[-1]*100:.1f}%, Final V: {voltages[-1]:.3f} V")
        
        return tte, results
    
    def quick_tte(self, power):
        """快速TTE估算（用于敏感性分析）"""
        tte, _ = self.simulate(power, dt=2.0, verbose=False)
        return tte

# ==================== 功率曲线生成器（确定性版本）====================
class PowerProfile:
    """确定性功率曲线生成器"""
    
    @staticmethod
    def constant(power_value):
        """恒定功率"""
        return lambda t: power_value
    
    @staticmethod
    def gaming(base_power=4.0):
        """游戏场景：高功率，周期性波动"""
        def power_func(t):
            t_h = t / 3600
            # 多频率叠加的确定性波动
            wave1 = 0.5 * np.sin(2 * np.pi * t / 30)   # 30秒周期
            wave2 = 0.3 * np.sin(2 * np.pi * t / 120)  # 2分钟周期
            wave3 = 0.2 * np.sin(2 * np.pi * t / 300)  # 5分钟周期
            # 温度效应
            temp = 0.05 * t_h
            return base_power + wave1 + wave2 + wave3 + temp
        return power_func
    
    @staticmethod
    def streaming(base_power=2.0):
        """流媒体场景：中等功率，较稳定"""
        def power_func(t):
            # 轻微波动
            wave = 0.15 * np.sin(2 * np.pi * t / 60)
            # 周期性缓冲
            buffer = 0.3 * (1 + np.sin(2 * np.pi * t / 300)) / 2 * (np.sin(2 * np.pi * t / 5) > 0.95)
            return base_power + wave
        return power_func
    
    @staticmethod
    def idle(base_power=0.5):
        """待机场景：低功率"""
        def power_func(t):
            wave = 0.05 * np.sin(2 * np.pi * t / 600)
            return base_power + wave
        return power_func

# ==================== 图1: OCV-SOC特性曲线 ====================
def plot_ocv_soc_curve():
    """绘制OCV-SOC特性曲线"""
    print("Generating OCV-SOC characteristic curve...")
    
    params = BatteryParameters()
    soc = np.linspace(0, 1, 500)
    ocv = params.ocv_from_soc(soc)
    
    # 计算导数
    d_ocv = np.gradient(ocv, soc)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：OCV-SOC曲线
    ax1 = axes[0]
    ax1.plot(soc * 100, ocv, color=SCIColors.blue, linewidth=2.5, label='7th-order polynomial')
    
    # 标注关键点
    key_points = [0, 0.2, 0.5, 0.8, 1.0]
    for s in key_points:
        v = params.ocv_from_soc(s)
        ax1.scatter(s * 100, v, color=SCIColors.red, s=80, zorder=5, edgecolors='white', linewidth=1.5)
        ax1.annotate(f'({s*100:.0f}%, {v:.2f}V)', 
                    xy=(s*100, v), xytext=(5, 10), 
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    ax1.set_xlabel('State of Charge (%)')
    ax1.set_ylabel('Open Circuit Voltage (V)')
    ax1.set_title('(a) OCV-SOC Characteristic')
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(3.2, 4.3)
    ax1.legend(loc='lower right')
    
    # 右图：导数曲线
    ax2 = axes[1]
    ax2.plot(soc * 100, d_ocv, color=SCIColors.orange, linewidth=2.5)
    ax2.fill_between(soc * 100, d_ocv, alpha=0.3, color=SCIColors.orange)
    
    ax2.set_xlabel('State of Charge (%)')
    ax2.set_ylabel('dOCV/dSOC (V/100%SOC)')
    ax2.set_title('(b) OCV Sensitivity to SOC')
    ax2.set_xlim(-5, 105)
    
    # 添加公式框
    formula = (r"$V_{OCV} = 66.94 \cdot SOC^7 - 248.74 \cdot SOC^6 + 371.91 \cdot SOC^5$" + "\n"
               r"$\quad\quad\quad - 291.52 \cdot SOC^4 + 132.66 \cdot SOC^3 - 36.39 \cdot SOC^2$" + "\n"
               r"$\quad\quad\quad + 6.07 \cdot SOC + 3.26$")
    
    fig.text(0.5, -0.02, formula, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('OCV-SOC Relationship: 7th-Order Polynomial Model', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = OUTPUT_DIR / 'Fig1_OCV_SOC_Characteristic.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'Fig1_OCV_SOC_Characteristic.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {filepath}")
    return filepath

# ==================== 图2: 功率P敏感性分析 ====================
def plot_power_sensitivity_analysis():
    """功率P的敏感性分析"""
    print("Generating Power Sensitivity Analysis...")
    
    params = BatteryParameters()
    model = SecondOrderRCModel(params)
    
    # 功率范围 (0.5W到10W)
    power_values = np.linspace(0.5, 10.0, 30)
    tte_values = []
    
    print("  Calculating TTE for different power levels...")
    for i, P in enumerate(power_values):
        if (i + 1) % 5 == 0:
            print(f"    Progress: {i+1}/{len(power_values)}")
        tte = model.quick_tte(P)
        tte_values.append(tte)
    
    tte_values = np.array(tte_values)
    
    # 理论TTE（基于能量守恒）
    energy_wh = params.nominal_capacity * params.nominal_voltage
    theoretical_tte = energy_wh / power_values
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) TTE vs 功率
    ax1 = axes[0, 0]
    ax1.plot(power_values, tte_values, color=SCIColors.blue, linewidth=2.5, marker='o', markersize=5, label='RC Model')
    ax1.plot(power_values, theoretical_tte, color=SCIColors.red, linewidth=2, linestyle='--', label='Theoretical (Energy/P)')
    ax1.fill_between(power_values, theoretical_tte * 0.8, theoretical_tte * 1.2, 
                     color=SCIColors.red, alpha=0.2, label='±20% Region')
    
    ax1.set_xlabel('Power Load, P (W)')
    ax1.set_ylabel('Time-to-Empty, TTE (Hours)')
    ax1.set_title('(a) TTE vs. Power Load')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 标注典型应用场景
    typical_powers = {
        'Idle': 0.5,
        'Streaming': 2.0,
        'Gaming': 4.0,
        'Heavy Load': 6.0,
        'Max Load': 8.0
    }
    
    for name, p in typical_powers.items():
        if p <= max(power_values):
            idx = np.argmin(np.abs(power_values - p))
            tte = tte_values[idx]
            ax1.annotate(f'{name}\n({p}W, {tte:.1f}h)', 
                        xy=(p, tte), xytext=(10, 10), 
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    
    # (b) 对数坐标下的TTE vs 功率
    ax2 = axes[0, 1]
    ax2.loglog(power_values, tte_values, color=SCIColors.green, linewidth=2.5, marker='s', markersize=5, label='RC Model')
    ax2.loglog(power_values, theoretical_tte, color=SCIColors.red, linewidth=2, linestyle='--', label='Theoretical')
    
    # 拟合幂律关系
    coeffs = np.polyfit(np.log(power_values), np.log(tte_values), 1)
    power_law_fit = np.exp(coeffs[1]) * power_values**coeffs[0]
    ax2.loglog(power_values, power_law_fit, color=SCIColors.purple, linewidth=2, 
               linestyle=':', label=f'Power Law: TTE ∝ P^{coeffs[0]:.2f}')
    
    ax2.set_xlabel('Power Load, P (W)')
    ax2.set_ylabel('Time-to-Empty, TTE (Hours)')
    ax2.set_title('(b) Log-Log Scale: TTE vs. Power')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6, which='both')
    
    # (c) 归一化TTE
    ax3 = axes[1, 0]
    tte_norm = tte_values / tte_values.max()
    theoretical_norm = theoretical_tte / theoretical_tte.max()
    
    ax3.plot(power_values, tte_norm, color=SCIColors.orange, linewidth=2.5, label='RC Model')
    ax3.plot(power_values, theoretical_norm, color=SCIColors.red, linewidth=2, linestyle='--', label='Theoretical')
    ax3.fill_between(power_values, tte_norm, theoretical_norm, 
                     where=abs(tte_norm - theoretical_norm) > 0.1, 
                     color=SCIColors.red, alpha=0.2, label='Difference > 10%')
    
    ax3.set_xlabel('Power Load, P (W)')
    ax3.set_ylabel('Normalized TTE (0-1)')
    ax3.set_title('(c) Normalized TTE Comparison')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # (d) 功率对TTE的导数（灵敏度）
    ax4 = axes[1, 1]
    d_tte_dP = np.gradient(tte_values, power_values)
    sensitivity = -d_tte_dP / tte_values * 100  # 百分比变化
    
    ax4.plot(power_values, sensitivity, color=SCIColors.cyan, linewidth=2.5)
    ax4.fill_between(power_values, sensitivity, 0, where=sensitivity>0, 
                     color=SCIColors.cyan, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 标注灵敏度范围
    ax4.axhline(y=np.mean(sensitivity), color=SCIColors.red, linestyle='--', 
               linewidth=1, label=f'Mean: {np.mean(sensitivity):.1f}%/W')
    ax4.fill_between(power_values, np.mean(sensitivity)-np.std(sensitivity), 
                     np.mean(sensitivity)+np.std(sensitivity), 
                     color=SCIColors.red, alpha=0.2, label='±1σ')
    
    ax4.set_xlabel('Power Load, P (W)')
    ax4.set_ylabel('TTE Sensitivity (%/W)')
    ax4.set_title('(d) TTE Sensitivity to Power Changes')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    # 添加统计信息
    stats_text = (f'Power Sensitivity Statistics:\n'
                  f'Mean Sensitivity: {np.mean(sensitivity):.2f} %/W\n'
                  f'Std Dev: {np.std(sensitivity):.2f} %/W\n'
                  f'Max Sensitivity: {np.max(sensitivity):.2f} %/W @ {power_values[np.argmax(sensitivity)]:.1f}W\n'
                  f'Min Sensitivity: {np.min(sensitivity):.2f} %/W @ {power_values[np.argmin(sensitivity)]:.1f}W')
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Power Load Sensitivity Analysis for TTE Prediction', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = OUTPUT_DIR / 'Fig2_Power_Sensitivity.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'Fig2_Power_Sensitivity.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {filepath}")
    return filepath

# ==================== 图3: 容量C敏感性分析 ====================
def plot_capacity_sensitivity_analysis():
    """容量C的敏感性分析"""
    print("Generating Capacity Sensitivity Analysis...")
    
    # 容量范围 (1Ah到5Ah)
    capacity_values = np.linspace(1.0, 5.0, 25)
    
    # 不同功率下的TTE
    power_levels = [1.0, 3.0, 5.0, 8.0]
    results = {p: [] for p in power_levels}
    
    print("  Calculating TTE for different capacities...")
    for i, capacity in enumerate(capacity_values):
        if (i + 1) % 5 == 0:
            print(f"    Progress: {i+1}/{len(capacity_values)}")
        
        params = BatteryParameters()
        params.nominal_capacity = capacity
        params.q_max = capacity * 3600
        model = SecondOrderRCModel(params)
        
        for p in power_levels:
            tte = model.quick_tte(p)
            results[p].append(tte)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = SCIColors.get_palette(len(power_levels))
    
    # (a) TTE vs 容量
    ax1 = axes[0, 0]
    for i, (p, ttes) in enumerate(results.items()):
        ax1.plot(capacity_values, ttes, color=colors[i], linewidth=2.5, 
                marker='o', markersize=5, label=f'P = {p} W')
    
    # 理论线性关系
    base_capacity = 3.0
    base_power = 3.0
    energy_per_ah = params.nominal_voltage  # 每Ah的能量
    theoretical_linear = capacity_values * energy_per_ah / base_power
    ax1.plot(capacity_values, theoretical_linear, color='black', linewidth=2, 
            linestyle='--', label='Linear Theory')
    
    ax1.set_xlabel('Battery Capacity, C (Ah)')
    ax1.set_ylabel('Time-to-Empty, TTE (Hours)')
    ax1.set_title('(a) TTE vs. Battery Capacity')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 标注典型电池容量
    typical_capacities = {
        'Smartwatch': 0.5,
        'Phone': 3.0,
        'Tablet': 8.0,
        'Laptop': 50.0,
        'EV': 500.0
    }
    
    # 只标注在当前范围内的容量
    for name, c in typical_capacities.items():
        if 1.0 <= c <= 5.0:
            idx = np.argmin(np.abs(capacity_values - c))
            ax1.axvline(x=c, color=SCIColors.gray, linestyle=':', alpha=0.5)
            ax1.annotate(name, xy=(c, ax1.get_ylim()[0]), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8, rotation=90)
    
    # (b) 归一化TTE
    ax2 = axes[0, 1]
    for i, (p, ttes) in enumerate(results.items()):
        ttes_norm = np.array(ttes) / np.array(ttes).max()
        ax2.plot(capacity_values, ttes_norm, color=colors[i], linewidth=2.5, 
                label=f'P = {p} W')
    
    # 线性归一化
    linear_norm = capacity_values / capacity_values.max()
    ax2.plot(capacity_values, linear_norm, color='black', linewidth=2, 
            linestyle='--', label='Ideal Linear')
    
    ax2.set_xlabel('Battery Capacity, C (Ah)')
    ax2.set_ylabel('Normalized TTE (0-1)')
    ax2.set_title('(b) Normalized TTE Comparison')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # (c) 容量对TTE的导数（灵敏度）
    ax3 = axes[1, 0]
    
    for i, (p, ttes) in enumerate(results.items()):
        d_tte_dC = np.gradient(ttes, capacity_values)
        sensitivity = d_tte_dC / np.array(ttes) * 100  # 百分比变化
        
        ax3.plot(capacity_values, sensitivity, color=colors[i], linewidth=2.5, 
                label=f'P = {p} W')
    
    ax3.axhline(y=100/3, color='black', linestyle='--', alpha=0.7, 
               label='Theoretical (33.3%/Ah)')
    ax3.set_xlabel('Battery Capacity, C (Ah)')
    ax3.set_ylabel('TTE Sensitivity (%/Ah)')
    ax3.set_title('(c) TTE Sensitivity to Capacity Changes')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # (d) 容量-功率联合分析
    ax4 = axes[1, 1]
    
    # 创建网格
    capacity_grid = np.linspace(1.0, 5.0, 20)
    power_grid = np.linspace(0.5, 8.0, 20)
    C_grid, P_grid = np.meshgrid(capacity_grid, power_grid)
    TTE_grid = np.zeros_like(C_grid)
    
    print("  Creating capacity-power surface...")
    for i in range(len(capacity_grid)):
        for j in range(len(power_grid)):
            params = BatteryParameters()
            params.nominal_capacity = capacity_grid[i]
            params.q_max = capacity_grid[i] * 3600
            model = SecondOrderRCModel(params)
            tte = model.quick_tte(power_grid[j])
            TTE_grid[j, i] = tte
    
    # 绘制等高线图
    contour = ax4.contourf(C_grid, P_grid, TTE_grid, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax4)
    cbar.set_label('TTE (Hours)', rotation=270, labelpad=15)
    
    # 添加等高线
    CS = ax4.contour(C_grid, P_grid, TTE_grid, levels=10, colors='white', linewidths=0.5, alpha=0.8)
    ax4.clabel(CS, inline=True, fontsize=8, fmt='%.1f h')
    
    ax4.set_xlabel('Battery Capacity, C (Ah)')
    ax4.set_ylabel('Power Load, P (W)')
    ax4.set_title('(d) Capacity-Power Trade-off Surface')
    ax4.grid(True, linestyle=':', alpha=0.3)
    
    # 添加典型设计点
    design_points = [
        (3.0, 3.0, 'Phone (3Ah, 3W)'),
        (4.0, 4.0, 'Tablet (4Ah, 4W)'),
        (5.0, 2.0, 'Long-life (5Ah, 2W)'),
        (2.0, 6.0, 'High-power (2Ah, 6W)')
    ]
    
    for c, p, label in design_points:
        ax4.scatter(c, p, color='red', s=80, edgecolors='white', linewidth=1.5, zorder=5)
        ax4.annotate(label, xy=(c, p), xytext=(10, 10), 
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 添加统计信息
    avg_sensitivity = {}
    for p, ttes in results.items():
        d_tte_dC = np.gradient(ttes, capacity_values)
        sensitivity = d_tte_dC / np.array(ttes) * 100
        avg_sensitivity[p] = np.mean(sensitivity)
    
    stats_text = (f'Capacity Sensitivity Summary:\n'
                  f'Average TTE increase per Ah:\n')
    for p, sens in avg_sensitivity.items():
        stats_text += f'  P={p}W: {sens:.1f}%/Ah\n'
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Battery Capacity Sensitivity Analysis for TTE Prediction', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = OUTPUT_DIR / 'Fig3_Capacity_Sensitivity.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'Fig3_Capacity_Sensitivity.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {filepath}")
    return filepath

# ==================== 图4: 蒙特卡洛不确定性分析 ====================
def plot_monte_carlo_analysis():
    """蒙特卡洛不确定性分析"""
    print("Generating Monte Carlo analysis...")
    
    np.random.seed(42)
    n_simulations = 500
    
    tte_results = []
    
    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_simulations}")
        
        # 随机参数（正态分布）
        params = BatteryParameters()
        params.R0 = max(0.02, np.random.normal(0.05, 0.01))
        params.R1 = max(0.005, np.random.normal(0.015, 0.003))
        params.R2 = max(0.01, np.random.normal(0.03, 0.006))
        params.nominal_capacity = max(2.5, np.random.normal(3.0, 0.15))
        params.q_max = params.nominal_capacity * 3600
        
        # 随机功率
        power = max(1.5, np.random.normal(3.0, 0.3))
        
        # 仿真
        model = SecondOrderRCModel(params)
        tte = model.quick_tte(power)
        tte_results.append(tte)
    
    tte_results = np.array(tte_results)
    
    # 统计
    mean_tte = np.mean(tte_results)
    std_tte = np.std(tte_results)
    ci_5 = np.percentile(tte_results, 5)
    ci_95 = np.percentile(tte_results, 95)
    median_tte = np.median(tte_results)
    
    print(f"  Mean TTE: {mean_tte:.3f} ± {std_tte:.3f} h")
    print(f"  90% CI: [{ci_5:.3f}, {ci_95:.3f}] h")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：直方图+KDE
    ax1 = axes[0]
    
    # 直方图
    n, bins, patches = ax1.hist(tte_results, bins=35, density=True, 
                                 color=SCIColors.blue, alpha=0.6, edgecolor='white', linewidth=0.8)
    
    # KDE曲线
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(tte_results)
    x_kde = np.linspace(tte_results.min(), tte_results.max(), 200)
    ax1.plot(x_kde, kde(x_kde), color=SCIColors.blue, linewidth=2.5, label='KDE')
    
    # 正态拟合
    x_norm = np.linspace(tte_results.min(), tte_results.max(), 200)
    y_norm = norm.pdf(x_norm, mean_tte, std_tte)
    ax1.plot(x_norm, y_norm, color=SCIColors.red, linewidth=2, linestyle='--',
            label=f'Normal fit ($\mu$={mean_tte:.2f}, $\sigma$={std_tte:.2f})')
    
    # 置信区间
    ax1.axvline(ci_5, color=SCIColors.orange, linewidth=2, linestyle='-', label='90% CI')
    ax1.axvline(ci_95, color=SCIColors.orange, linewidth=2, linestyle='-')
    ax1.axvline(median_tte, color=SCIColors.green, linewidth=2, linestyle=':', 
               label=f'Median: {median_tte:.2f}h')
    
    # 填充置信区间
    ax1.axvspan(ci_5, ci_95, alpha=0.15, color=SCIColors.orange)
    
    ax1.set_xlabel('Predicted Time-to-Empty (Hours)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('(a) TTE Distribution')
    ax1.legend(loc='upper right', fontsize=9)
    
    # 右图：累积分布
    ax2 = axes[1]
    
    sorted_tte = np.sort(tte_results)
    cdf = np.arange(1, len(sorted_tte) + 1) / len(sorted_tte)
    
    ax2.plot(sorted_tte, cdf * 100, color=SCIColors.blue, linewidth=2.5)
    ax2.fill_between(sorted_tte, cdf * 100, alpha=0.3, color=SCIColors.blue)
    
    # 标注百分位
    for pct in [5, 25, 50, 75, 95]:
        val = np.percentile(tte_results, pct)
        ax2.axhline(pct, color=SCIColors.gray, linewidth=0.8, linestyle='--', alpha=0.5)
        ax2.axvline(val, color=SCIColors.gray, linewidth=0.8, linestyle='--', alpha=0.5)
        ax2.scatter(val, pct, color=SCIColors.red, s=50, zorder=5)
        ax2.annotate(f'{pct}%: {val:.2f}h', xy=(val, pct), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Time-to-Empty (Hours)')
    ax2.set_ylabel('Cumulative Probability (%)')
    ax2.set_title('(b) Cumulative Distribution Function')
    ax2.set_ylim(0, 105)
    
    # 统计信息框
    stats_text = (f'Statistics (N={n_simulations}):\n'
                  f'Mean: {mean_tte:.3f} h\n'
                  f'Std: {std_tte:.3f} h\n'
                  f'Median: {median_tte:.3f} h\n'
                  f'90% CI: [{ci_5:.2f}, {ci_95:.2f}] h')
    
    ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Monte Carlo Uncertainty Analysis of TTE Prediction', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = OUTPUT_DIR / 'Fig4_MonteCarlo_Uncertainty.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'Fig4_MonteCarlo_Uncertainty.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {filepath}")
    return filepath

# ==================== 图5: 场景对比分析 ====================
def plot_scenario_comparison():
    """场景对比分析"""
    print("Generating scenario comparison...")
    
    params = BatteryParameters()
    model = SecondOrderRCModel(params)
    
    # 定义场景
    scenarios = {
        'Gaming': {'power_func': PowerProfile.gaming(4.0), 'color': SCIColors.red},
        'Streaming': {'power_func': PowerProfile.streaming(2.0), 'color': SCIColors.blue},
        'Constant 3W': {'power_func': PowerProfile.constant(3.0), 'color': SCIColors.green},
        'Idle': {'power_func': PowerProfile.idle(0.5), 'color': SCIColors.purple},
    }
    
    results = {}
    
    for name, config in scenarios.items():
        print(f"  Simulating: {name}")
        tte, data = model.simulate(config['power_func'], dt=1.0, verbose=False)
        results[name] = {'tte': tte, 'data': data, 'color': config['color']}
        print(f"    TTE: {tte:.2f} h")
    
    # 绘图 (2x2布局)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) 电压响应
    ax1 = axes[0, 0]
    for name, r in results.items():
        time_h = r['data']['time'] / 3600
        ax1.plot(time_h, r['data']['voltage'], color=r['color'], linewidth=2,
                label=f"{name} (TTE: {r['tte']:.2f}h)")
    
    ax1.axhline(params.cutoff_voltage, color=SCIColors.gray, linestyle='--', 
               linewidth=1.5, label=f'Cutoff: {params.cutoff_voltage}V')
    ax1.set_xlabel('Time (Hours)')
    ax1.set_ylabel('Terminal Voltage (V)')
    ax1.set_title('(a) Voltage Response')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_ylim(2.9, 4.3)
    
    # (b) SOC曲线
    ax2 = axes[0, 1]
    for name, r in results.items():
        time_h = r['data']['time'] / 3600
        ax2.plot(time_h, r['data']['soc'] * 100, color=r['color'], linewidth=2, label=name)
    
    ax2.set_xlabel('Time (Hours)')
    ax2.set_ylabel('State of Charge (%)')
    ax2.set_title('(b) SOC Discharge Curves')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(-5, 105)
    
    # (c) 功率曲线
    ax3 = axes[1, 0]
    t_plot = np.linspace(0, 4 * 3600, 500)
    for name, config in scenarios.items():
        if name != 'Idle':  # 跳过Idle场景，功率太低
            power_values = [config['power_func'](t) for t in t_plot]
            ax3.plot(t_plot / 3600, power_values, color=results[name]['color'], 
                    linewidth=2, label=name, alpha=0.8)
    
    ax3.set_xlabel('Time (Hours)')
    ax3.set_ylabel('Power Consumption (W)')
    ax3.set_title('(c) Power Profiles')
    ax3.legend(loc='upper right', fontsize=9)
    
    # (d) TTE比较条形图
    ax4 = axes[1, 1]
    names = list(results.keys())
    ttes = [results[n]['tte'] for n in names]
    colors = [results[n]['color'] for n in names]
    
    bars = ax4.bar(names, ttes, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for bar, tte in zip(bars, ttes):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{tte:.2f}h', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_ylabel('Time-to-Empty (Hours)')
    ax4.set_title('(d) TTE Comparison')
    ax4.set_ylim(0, max(ttes) * 1.15)
    
    plt.suptitle('Scenario Analysis: Discharge Behavior Under Different Power Profiles',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = OUTPUT_DIR / 'Fig5_Scenario_Comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'Fig5_Scenario_Comparison.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {filepath}")
    return filepath

# ==================== 图6: 综合模型验证 ====================
def plot_model_validation():
    """模型验证：展示RC动态响应"""
    print("Generating model validation plot...")
    
    params = BatteryParameters()
    model = SecondOrderRCModel(params)
    
    # 阶跃功率响应
    def step_power(t):
        if t < 600:
            return 1.0
        elif t < 1200:
            return 4.0
        elif t < 1800:
            return 2.0
        else:
            return 3.0
    
    print("  Simulating step response...")
    tte, data = model.simulate(step_power, t_max=3600, dt=0.5, verbose=False)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    time_min = data['time'] / 60
    
    # (a) 功率输入
    ax1 = axes[0, 0]
    power_values = [step_power(t) for t in data['time']]
    ax1.step(time_min, power_values, color=SCIColors.red, linewidth=2, where='post')
    ax1.fill_between(time_min, power_values, step='post', alpha=0.3, color=SCIColors.red)
    ax1.set_xlabel('Time (Minutes)')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('(a) Step Power Input')
    ax1.set_ylim(0, 5)
    
    # (b) 电压响应
    ax2 = axes[0, 1]
    ax2.plot(time_min, data['voltage'], color=SCIColors.blue, linewidth=2, label='Terminal Voltage')
    
    # 计算OCV
    ocv = params.ocv_from_soc(data['soc'])
    ax2.plot(time_min, ocv, color=SCIColors.green, linewidth=2, linestyle='--', label='OCV')
    
    ax2.set_xlabel('Time (Minutes)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_title('(b) Voltage Response')
    ax2.legend(loc='upper right')
    
    # (c) RC极化电压
    ax3 = axes[1, 0]
    ax3.plot(time_min, data['v1'] * 1000, color=SCIColors.orange, linewidth=2, label='$V_1$ (RC1)')
    ax3.plot(time_min, data['v2'] * 1000, color=SCIColors.purple, linewidth=2, label='$V_2$ (RC2)')
    ax3.plot(time_min, (data['v1'] + data['v2']) * 1000, color=SCIColors.gray, linewidth=2, 
            linestyle='--', label='$V_1 + V_2$')
    
    ax3.set_xlabel('Time (Minutes)')
    ax3.set_ylabel('Polarization Voltage (mV)')
    ax3.set_title('(c) RC Polarization Dynamics')
    ax3.legend(loc='upper right')
    
    # (d) 电流响应
    ax4 = axes[1, 1]
    ax4.plot(time_min, data['current'], color=SCIColors.cyan, linewidth=2)
    ax4.fill_between(time_min, data['current'], alpha=0.3, color=SCIColors.cyan)
    ax4.set_xlabel('Time (Minutes)')
    ax4.set_ylabel('Current (A)')
    ax4.set_title('(d) Current Response')
    
    # 添加参数框
    param_text = (f"Model Parameters:\n"
                  f"$R_0$ = {params.R0*1000:.0f} mΩ\n"
                  f"$R_1$ = {params.R1*1000:.0f} mΩ, $C_1$ = {params.C1:.0f} F\n"
                  f"$R_2$ = {params.R2*1000:.0f} mΩ, $C_2$ = {params.C2:.0f} F\n"
                  f"$τ_1$ = {params.R1*params.C1:.0f} s, $τ_2$ = {params.R2*params.C2:.0f} s")
    
    fig.text(0.02, 0.02, param_text, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Model Validation: 2nd-Order RC Dynamic Response to Step Power',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = OUTPUT_DIR / 'Fig6_Model_Validation.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'Fig6_Model_Validation.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {filepath}")
    return filepath

# ==================== 主程序 ====================
def main():
    """主程序入口"""
    print("=" * 70)
    print("  2nd-Order RC Battery Model Simulation")
    print("  with 7th-Order OCV-SOC Polynomial")
    print("=" * 70)
    
    # 显示模型参数
    params = BatteryParameters()
    print("\n[Model Parameters]")
    print(f"  Battery Capacity: {params.nominal_capacity} Ah")
    print(f"  Nominal Voltage: {params.nominal_voltage} V")
    print(f"  Cutoff Voltage: {params.cutoff_voltage} V")
    print(f"  R0: {params.R0*1000:.1f} mΩ")
    print(f"  R1: {params.R1*1000:.1f} mΩ, C1: {params.C1} F (τ1 = {params.R1*params.C1:.0f} s)")
    print(f"  R2: {params.R2*1000:.1f} mΩ, C2: {params.C2} F (τ2 = {params.R2*params.C2:.0f} s)")
    
    print(f"\n[Output Directory]: {OUTPUT_DIR.absolute()}")
    
    # 生成所有图形
    print("\n" + "=" * 70)
    print("  Generating Figures...")
    print("=" * 70 + "\n")
    
    figures = []
    
    # 图1: OCV-SOC曲线
    figures.append(plot_ocv_soc_curve())
    
    # 图2: 功率P敏感性分析
    figures.append(plot_power_sensitivity_analysis())
    
    # 图3: 容量C敏感性分析
    figures.append(plot_capacity_sensitivity_analysis())
    
    # 图4: 蒙特卡洛分析
    figures.append(plot_monte_carlo_analysis())
    
    # 图5: 场景对比
    figures.append(plot_scenario_comparison())
    
    # 图6: 模型验证
    figures.append(plot_model_validation())
    
    # 总结
    print("\n" + "=" * 70)
    print("  All figures generated successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    for f in figures:
        print(f"  ✓ {f}")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nNote: Both PNG (300 dpi) and PDF versions are saved for each figure.")
    print("      PDF format is recommended for journal submission.")

if __name__ == "__main__":
    main()