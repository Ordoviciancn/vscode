import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from enum import Enum
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================
# 1. 设置SCI论文图表样式
# =============================================

def set_sci_style():
    """设置SCI论文图表样式"""
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        
        # 图表尺寸
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        
        # 线条设置
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # 坐标轴设置
        'axes.linewidth': 1.2,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.grid': True,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        
        # 刻度设置
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        
        # 图例设置
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'black',
        
        # 图像设置
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

# 应用样式
set_sci_style()

# =============================================
# 2. 数据类型定义
# =============================================

class SolverMethod(Enum):
    """求解方法枚举"""
    EULER = "euler"
    RK4 = "rk4"

@dataclass
class BatteryParams:
    """电池模型参数"""
    # 电池基本信息
    name: str = "Li-ion_Battery"
    nominal_voltage: float = 3.7  # 标称电压 (V)
    
    # 电池容量参数
    Cn: float = 3.0  # 额定容量 (Ah)
    SOH: float = 0.95  # 健康状态 (0-1)
    eta: float = 0.99  # 库仑效率
    
    # 等效电路模型参数
    R0: float = 0.05  # 欧姆内阻 (Ω)
    R1: float = 0.02  # 第一极化电阻 (Ω)
    C1: float = 2000  # 第一极化电容 (F)
    R2: float = 0.03  # 第二极化电阻 (Ω)
    C2: float = 10000  # 第二极化电容 (F)
    
    # 功率转换效率
    eta_conv: float = 0.95  # 功率转换效率
    
    # 初始条件
    SOC0: float = 1.0  # 初始SOC (0-1)
    Vp10: float = 0.0  # 初始极化电压1 (V)
    Vp20: float = 0.0  # 初始极化电压2 (V)
    
    # 终止条件
    SOC_min: float = 0.05  # 最小SOC
    V_min: float = 3.2  # 最小电压 (V)
    t_max: float = 36000  # 最大模拟时间 (s) - 10小时
    
    # OCV-SOC关系参数
    ocv_coeffs: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        # 如果没有提供OCV系数，使用默认的七阶多项式
        if not self.ocv_coeffs:
            # 七阶多项式系数
            self.ocv_coeffs = [3.259, 6.069, -36.387, 132.659, -291.517, 
                             371.911, -248.739, 66.939]
        
        # 计算时间常数
        self.tau1 = self.R1 * self.C1
        self.tau2 = self.R2 * self.C2

@dataclass
class SimulationResults:
    """存储仿真结果"""
    time: List[float] = field(default_factory=list)
    SOC: List[float] = field(default_factory=list)
    Vt: List[float] = field(default_factory=list)
    I: List[float] = field(default_factory=list)
    V_oc: List[float] = field(default_factory=list)
    V_p1: List[float] = field(default_factory=list)
    V_p2: List[float] = field(default_factory=list)
    P: List[float] = field(default_factory=list)
    termination_reason: str = ""
    termination_time: float = 0.0
    
    def add_point(self, t: float, SOC_val: float, Vt_val: float, I_val: float,
                  V_oc_val: float, V_p1_val: float, V_p2_val: float, 
                  P_val: float):
        """添加一个时间点的数据"""
        self.time.append(t)
        self.SOC.append(SOC_val)
        self.Vt.append(Vt_val)
        self.I.append(I_val)
        self.V_oc.append(V_oc_val)
        self.V_p1.append(V_p1_val)
        self.V_p2.append(V_p2_val)
        self.P.append(P_val)
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        df = pd.DataFrame({
            'Time(s)': self.time,
            'Time(h)': np.array(self.time) / 3600,
            'SOC': self.SOC,
            'SOC(%)': np.array(self.SOC) * 100,
            'Voltage(V)': self.Vt,
            'Current(A)': self.I,
            'OCV(V)': self.V_oc,
            'V_p1(V)': self.V_p1,
            'V_p2(V)': self.V_p2,
            'Power(W)': self.P,
        })
        
        # 计算能量和容量
        if len(self.time) > 1:
            time_array = np.array(self.time)
            current_array = np.array(self.I)
            power_array = np.array(self.P)
            
            # 使用梯形法计算积分
            dt = np.diff(time_array)
            energy = np.cumsum(0.5 * (power_array[1:] + power_array[:-1]) * dt) / 3600
            capacity = np.cumsum(0.5 * (current_array[1:] + current_array[:-1]) * dt) / 3600
            
            df['Energy(Wh)'] = np.concatenate([[0], energy])
            df['Capacity(Ah)'] = np.concatenate([[0], capacity])
        
        return df
    
    def get_summary(self) -> Dict:
        """获取仿真摘要"""
        if not self.time:
            return {}
        
        # 手动计算梯形积分
        def trapezoidal_integral(y, x):
            if len(y) != len(x) or len(y) <= 1:
                return 0.0
            integral = 0.0
            for i in range(len(x) - 1):
                integral += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
            return integral
        
        total_capacity = trapezoidal_integral(self.I, self.time)
        total_energy = trapezoidal_integral(self.P, self.time)
        
        return {
            'termination_reason': self.termination_reason,
            'termination_time_s': self.termination_time,
            'termination_time_h': self.termination_time / 3600,
            'final_SOC': self.SOC[-1],
            'final_voltage': self.Vt[-1],
            'total_capacity_Ah': total_capacity / 3600,
            'total_energy_Wh': total_energy / 3600,
            'max_current': max(self.I) if self.I else 0,
            'min_current': min(self.I) if self.I else 0,
            'max_voltage': max(self.Vt) if self.Vt else 0,
            'min_voltage': min(self.Vt) if self.Vt else 0,
            'average_power': np.mean(self.P) if self.P else 0
        }

# =============================================
# 3. 电池模型核心类
# =============================================

class BatteryModel:
    """电池模型核心类"""
    
    def __init__(self, params: BatteryParams):
        self.params = params
        self.ocv_func = self._create_ocv_function()
    
    def _create_ocv_function(self) -> Callable:
        """创建OCV-SOC函数"""
        coeffs = self.params.ocv_coeffs
        return lambda soc: self._poly_ocv(soc, coeffs)
    
    def _poly_ocv(self, soc: float, coeffs: List[float]) -> float:
        """多项式计算OCV"""
        soc_clipped = max(min(soc, 1.0), 0.0)
        ocv = coeffs[0]
        for i in range(1, len(coeffs)):
            ocv += coeffs[i] * (soc_clipped ** i)
        return ocv
    
    def calculate_vt(self, SOC: float, Vp1: float, Vp2: float, 
                    P0: float) -> Tuple[float, float, bool]:
        """
        计算端电压和电流
        返回: (V_t, I, feasible)
        """
        # 计算OCV
        V_oc = self.ocv_func(SOC)
        
        # 解一元二次方程: V_t^2 - b*V_t + c = 0
        b = V_oc - Vp1 - Vp2
        c = P0 * self.params.R0 / self.params.eta_conv
        
        discriminant = b**2 - 4 * c
        
        if discriminant < 0:
            return 0.0, 0.0, False
        
        # 取物理合理的正根 (较高电压解)
        V_t = (b + np.sqrt(discriminant)) / 2.0
        I = P0 / (self.params.eta_conv * V_t)
        
        return V_t, I, True
    
    def calculate_derivatives(self, state: np.ndarray, P0: float) -> np.ndarray:
        """
        计算状态变量的导数
        state = [SOC, Vp1, Vp2]
        返回: dstate/dt = [dSOC/dt, dVp1/dt, dVp2/dt]
        """
        SOC, Vp1, Vp2 = state
        
        # 计算端电压和电流
        V_t, I, feasible = self.calculate_vt(SOC, Vp1, Vp2, P0)
        
        if not feasible:
            return np.zeros(3)
        
        # 计算导数
        dSOC = -self.params.eta * I / (self.params.Cn * self.params.SOH * 3600)
        dVp1 = I / self.params.C1 - Vp1 / (self.params.R1 * self.params.C1)
        dVp2 = I / self.params.C2 - Vp2 / (self.params.R2 * self.params.C2)
        
        return np.array([dSOC, dVp1, dVp2])

class BatterySolver:
    """电池模型求解器"""
    
    def __init__(self, params: BatteryParams):
        self.params = params
        self.model = BatteryModel(params)
    
    def solve_constant_power(self, P0: float, method: SolverMethod = SolverMethod.RK4,
                            dt: float = 1.0) -> SimulationResults:
        """
        求解恒定功耗下的电池放电过程
        """
        results = SimulationResults()
        
        # 初始化状态变量
        t = 0.0
        SOC = self.params.SOC0
        Vp1 = self.params.Vp10
        Vp2 = self.params.Vp20
        
        # 最大迭代次数保护
        max_iterations = int(self.params.t_max / dt) + 1000
        iteration = 0
        
        while True:
            iteration += 1
            if iteration > max_iterations:
                results.termination_reason = "Maximum iterations reached"
                break
            
            # 计算当前开路电压
            V_oc = self.model.ocv_func(SOC)
            
            # 计算端电压和电流
            V_t, I, feasible = self.model.calculate_vt(SOC, Vp1, Vp2, P0)
            
            if not feasible:
                results.termination_reason = f"Power demand {P0}W exceeds maximum deliverable power"
                results.termination_time = t
                break
            
            # 记录当前结果
            results.add_point(t, SOC, V_t, I, V_oc, Vp1, Vp2, P0)
            
            # 检查终止条件
            if V_t <= self.params.V_min:
                results.termination_time = t
                results.termination_reason = f"Voltage reached cutoff {self.params.V_min}V"
                break
            
            if SOC <= self.params.SOC_min:
                results.termination_time = t
                results.termination_reason = f"SOC reached minimum {self.params.SOC_min}"
                break
            
            if t >= self.params.t_max:
                results.termination_time = t
                results.termination_reason = f"Maximum simulation time {self.params.t_max}s reached"
                break
            
            # 更新状态变量
            if method == SolverMethod.EULER:
                SOC, Vp1, Vp2, t = self._euler_step(SOC, Vp1, Vp2, t, P0, dt)
            elif method == SolverMethod.RK4:
                SOC, Vp1, Vp2, t = self._rk4_step(SOC, Vp1, Vp2, t, P0, dt)
        
        return results
    
    def _euler_step(self, SOC: float, Vp1: float, Vp2: float, t: float,
                   P0: float, dt: float) -> Tuple[float, float, float, float]:
        """欧拉法单步更新"""
        state = np.array([SOC, Vp1, Vp2])
        derivatives = self.model.calculate_derivatives(state, P0)
        
        SOC_new = SOC + derivatives[0] * dt
        Vp1_new = Vp1 + derivatives[1] * dt
        Vp2_new = Vp2 + derivatives[2] * dt
        
        return SOC_new, Vp1_new, Vp2_new, t + dt
    
    def _rk4_step(self, SOC: float, Vp1: float, Vp2: float, t: float,
                 P0: float, dt: float) -> Tuple[float, float, float, float]:
        """RK4法单步更新"""
        state = np.array([SOC, Vp1, Vp2])
        
        k1 = self.model.calculate_derivatives(state, P0)
        k2 = self.model.calculate_derivatives(state + 0.5 * dt * k1, P0)
        k3 = self.model.calculate_derivatives(state + 0.5 * dt * k2, P0)
        k4 = self.model.calculate_derivatives(state + dt * k3, P0)
        
        state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return state_new[0], state_new[1], state_new[2], t + dt

# =============================================
# 4. SCI论文图表生成
# =============================================

class BatteryPaperFigures:
    """生成SCI论文图表的类"""
    
    def __init__(self):
        # 创建输出目录
        self.output_dir = Path("paper_figures")
        self.output_dir.mkdir(exist_ok=True)
        
        # 颜色方案 (适合黑白印刷)
        self.colors = {
            'primary': ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6'],
            'secondary': ['#95a5a6', '#7f8c8d', '#34495e', '#16a085', '#8e44ad'],
        }
    
    def figure1_detailed_discharge(self, battery_params: BatteryParams, 
                                   P0: float = 2.5):
        """
        图1: 详细放电曲线 (2.5W恒定功率)
        包含电压、SOC、电流、极化电压等
        """
        solver = BatterySolver(battery_params)
        results = solver.solve_constant_power(P0, SolverMethod.RK4, dt=0.5)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        df = results.to_dataframe()
        
        # 子图1: 电压和SOC曲线
        ax1 = axes[0, 0]
        ax1.plot(df['Time(h)'], df['Voltage(V)'], 
                color=self.colors['primary'][0], linewidth=2, label='Terminal Voltage')
        ax1.plot(df['Time(h)'], df['OCV(V)'], '--',
                color=self.colors['primary'][1], linewidth=1.5, alpha=0.7, label='OCV')
        ax1.axhline(y=battery_params.V_min, color='r', linestyle=':', 
                   linewidth=1.5, label='Cutoff Voltage')
        
        ax1.set_xlabel('Time (h)', fontweight='bold')
        ax1.set_ylabel('Voltage (V)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        
        # SOC次坐标轴
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df['Time(h)'], df['SOC(%)'], 
                     color=self.colors['primary'][2], linewidth=2, alpha=0.7)
        ax1_twin.set_ylabel('SOC (%)', fontweight='bold', color=self.colors['primary'][2])
        ax1_twin.tick_params(axis='y', labelcolor=self.colors['primary'][2])
        
        # 子图2: 电流和功率曲线
        ax2 = axes[0, 1]
        ax2.plot(df['Time(h)'], df['Current(A)'], 
                color=self.colors['primary'][3], linewidth=2, label='Current')
        ax2.set_xlabel('Time (h)', fontweight='bold')
        ax2.set_ylabel('Current (A)', fontweight='bold', color=self.colors['primary'][3])
        ax2.tick_params(axis='y', labelcolor=self.colors['primary'][3])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # 功率次坐标轴
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df['Time(h)'], df['Power(W)'], '--',
                     color=self.colors['primary'][4], linewidth=2, alpha=0.7, label='Power')
        ax2_twin.set_ylabel('Power (W)', fontweight='bold', color=self.colors['primary'][4])
        ax2_twin.tick_params(axis='y', labelcolor=self.colors['primary'][4])
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 子图3: 极化电压
        ax3 = axes[1, 0]
        ax3.plot(df['Time(h)'], df['V_p1(V)'], 
                color=self.colors['secondary'][0], linewidth=2, label='$V_{p1}$')
        ax3.plot(df['Time(h)'], df['V_p2(V)'], 
                color=self.colors['secondary'][1], linewidth=2, label='$V_{p2}$')
        ax3.plot(df['Time(h)'], df['V_p1(V)'] + df['V_p2(V)'], '--',
                color=self.colors['secondary'][2], linewidth=1.5, alpha=0.7, label='Total')
        
        ax3.set_xlabel('Time (h)', fontweight='bold')
        ax3.set_ylabel('Polarization Voltage (V)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        
        # 子图4: 累计能量和容量
        ax4 = axes[1, 1]
        if 'Energy(Wh)' in df.columns and 'Capacity(Ah)' in df.columns:
            ax4.plot(df['Time(h)'], df['Energy(Wh)'], 
                    color=self.colors['secondary'][3], linewidth=2, label='Energy')
            ax4.set_xlabel('Time (h)', fontweight='bold')
            ax4.set_ylabel('Energy (Wh)', fontweight='bold', color=self.colors['secondary'][3])
            ax4.tick_params(axis='y', labelcolor=self.colors['secondary'][3])
            ax4.grid(True, alpha=0.3)
            
            # 容量次坐标轴
            ax4_twin = ax4.twinx()
            ax4_twin.plot(df['Time(h)'], df['Capacity(Ah)'], '--',
                         color=self.colors['secondary'][4], linewidth=2, alpha=0.7, label='Capacity')
            ax4_twin.set_ylabel('Capacity (Ah)', fontweight='bold', color=self.colors['secondary'][4])
            ax4_twin.tick_params(axis='y', labelcolor=self.colors['secondary'][4])
            
            # 合并图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 添加标题和调整布局
        fig.suptitle(f'Constant Power Discharge: {P0} W', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片
        fig.savefig(self.output_dir / f'figure1_detailed_discharge_{P0}W.png', dpi=300)
        fig.savefig(self.output_dir / f'figure1_detailed_discharge_{P0}W.pdf')
        
        return fig, results
    
    def figure2_voltage_composition(self, battery_params: BatteryParams, 
                                   P0: float = 2.5):
        """
        图2: 电压组成分解图
        展示OCV、IR压降、极化压降之间的关系
        """
        solver = BatterySolver(battery_params)
        results = solver.solve_constant_power(P0, SolverMethod.RK4, dt=0.5)
        df = results.to_dataframe()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 计算各电压分量
        V_ir = df['Current(A)'] * battery_params.R0
        V_polar = df['V_p1(V)'] + df['V_p2(V)']
        V_ocv = df['OCV(V)']
        
        # 创建堆叠区域图
        ax.fill_between(df['Time(h)'], 0, V_ir, alpha=0.4, label='IR Drop', 
                       color=self.colors['primary'][1])
        ax.fill_between(df['Time(h)'], V_ir, V_ir + V_polar, alpha=0.4, 
                       label='Polarization', color=self.colors['primary'][2])
        ax.fill_between(df['Time(h)'], V_ir + V_polar, V_ocv, alpha=0.4, 
                       label='Available OCV', color=self.colors['primary'][3])
        
        # 绘制OCV和端电压曲线
        ax.plot(df['Time(h)'], df['OCV(V)'], 'k-', linewidth=2, label='OCV')
        ax.plot(df['Time(h)'], df['Voltage(V)'], 'r--', linewidth=2.5, label='Terminal Voltage')
        
        ax.set_xlabel('Time (h)', fontweight='bold')
        ax.set_ylabel('Voltage (V)', fontweight='bold')
        ax.set_title('Voltage Composition Analysis', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加标注
        midpoint_idx = len(df) // 2
        mid_time = df['Time(h)'].iloc[midpoint_idx]
        ax.annotate('IR Drop', xy=(mid_time, V_ir.iloc[midpoint_idx]/2),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=9, ha='left', va='center')
        ax.annotate('Polarization', xy=(mid_time, V_ir.iloc[midpoint_idx] + V_polar.iloc[midpoint_idx]/2),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=9, ha='left', va='center')
        
        plt.tight_layout()
        
        # 保存图片
        fig.savefig(self.output_dir / f'figure2_voltage_composition_{P0}W.png', dpi=300)
        fig.savefig(self.output_dir / f'figure2_voltage_composition_{P0}W.pdf')
        
        return fig
    
    def figure3_multiple_power_comparison(self, battery_params: BatteryParams):
        """
        图3: 不同功率下的放电曲线比较
        """
        power_levels = [1.0, 2.0, 3.0, 4.0]
        results_list = []
        
        # 运行不同功率的仿真
        for P in power_levels:
            solver = BatterySolver(battery_params)
            results = solver.solve_constant_power(P, SolverMethod.RK4, dt=0.5)
            results_list.append((f'{P} W', results))
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # 创建颜色映射
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(power_levels)))
        
        for idx, (label, results) in enumerate(results_list):
            df = results.to_dataframe()
            color = colors[idx]
            
            # 子图1: 电压曲线
            axes[0, 0].plot(df['Time(h)'], df['Voltage(V)'], 
                           color=color, linewidth=2, label=label)
            
            # 子图2: SOC曲线
            axes[0, 1].plot(df['Time(h)'], df['SOC(%)'], 
                           color=color, linewidth=2, label=label)
            
            # 子图3: 电流曲线
            axes[1, 0].plot(df['Time(h)'], df['Current(A)'], 
                           color=color, linewidth=2, label=label)
            
            # 子图4: 功率曲线
            axes[1, 1].plot(df['Time(h)'], df['Power(W)'], 
                           color=color, linewidth=2, label=label)
        
        # 设置子图属性
        axes[0, 0].set_xlabel('Time (h)', fontweight='bold')
        axes[0, 0].set_ylabel('Terminal Voltage (V)', fontweight='bold')
        axes[0, 0].set_title('Voltage vs Time', fontsize=12)
        axes[0, 0].legend(loc='best', fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Time (h)', fontweight='bold')
        axes[0, 1].set_ylabel('SOC (%)', fontweight='bold')
        axes[0, 1].set_title('SOC vs Time', fontsize=12)
        axes[0, 1].legend(loc='best', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Time (h)', fontweight='bold')
        axes[1, 0].set_ylabel('Current (A)', fontweight='bold')
        axes[1, 0].set_title('Current vs Time', fontsize=12)
        axes[1, 0].legend(loc='best', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Time (h)', fontweight='bold')
        axes[1, 1].set_ylabel('Power (W)', fontweight='bold')
        axes[1, 1].set_title('Power vs Time', fontsize=12)
        axes[1, 1].legend(loc='best', fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle('Constant Power Discharge at Different Power Levels', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片
        fig.savefig(self.output_dir / 'figure3_power_comparison.png', dpi=300)
        fig.savefig(self.output_dir / 'figure3_power_comparison.pdf')
        
        return fig, results_list
    
    def figure4_power_sweep_analysis(self, battery_params: BatteryParams):
        """
        图4: 功耗扫描分析
        展示放电时间、总能量、平均电流等与功率的关系
        """
        power_range = np.linspace(0.5, 6.0, 12)  # 从0.5W到6W
        summary_data = []
        
        print("Running power sweep analysis...")
        for P in power_range:
            solver = BatterySolver(battery_params)
            results = solver.solve_constant_power(P, SolverMethod.RK4, dt=0.5)
            summary = results.get_summary()
            summary['Power(W)'] = P
            summary_data.append(summary)
        
        df_summary = pd.DataFrame(summary_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # 子图1: 放电时间 vs 功率
        axes[0, 0].plot(df_summary['Power(W)'], df_summary['termination_time_h'], 
                       'o-', color=self.colors['primary'][0], linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Constant Power (W)', fontweight='bold')
        axes[0, 0].set_ylabel('Discharge Time (h)', fontweight='bold')
        axes[0, 0].set_title('Discharge Time vs Power', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数据点标注
        for _, row in df_summary.iterrows():
            axes[0, 0].annotate(f"{row['termination_time_h']:.1f}h", 
                              xy=(row['Power(W)'], row['termination_time_h']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8)
        
        # 子图2: 总能量 vs 功率
        axes[0, 1].plot(df_summary['Power(W)'], df_summary['total_energy_Wh'], 
                       's-', color=self.colors['primary'][1], linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Constant Power (W)', fontweight='bold')
        axes[0, 1].set_ylabel('Total Energy (Wh)', fontweight='bold')
        axes[0, 1].set_title('Total Energy vs Power', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: 平均电流 vs 功率
        axes[1, 0].plot(df_summary['Power(W)'], df_summary['average_power'] / battery_params.nominal_voltage,
                       '^-', color=self.colors['primary'][2], linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Constant Power (W)', fontweight='bold')
        axes[1, 0].set_ylabel('Average Current (A)', fontweight='bold')
        axes[1, 0].set_title('Average Current vs Power', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 效率 vs 功率
        if 'final_SOC' in df_summary.columns:
            theoretical_energy = df_summary['total_capacity_Ah'] * battery_params.nominal_voltage
            actual_energy = df_summary['total_energy_Wh']
            efficiency = np.where(theoretical_energy > 0, 
                                 actual_energy / theoretical_energy * 100, 0)
            
            axes[1, 1].plot(df_summary['Power(W)'], efficiency, 
                           'D-', color=self.colors['primary'][3], linewidth=2, markersize=6)
            axes[1, 1].axhline(y=battery_params.eta_conv * 100, color='r', linestyle='--',
                             label=f'η_conv={battery_params.eta_conv*100:.1f}%')
            axes[1, 1].set_xlabel('Constant Power (W)', fontweight='bold')
            axes[1, 1].set_ylabel('Discharge Efficiency (%)', fontweight='bold')
            axes[1, 1].set_title('Efficiency vs Power', fontsize=12)
            axes[1, 1].legend(loc='best', fontsize=9)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 105])
        
        fig.suptitle('Power Sweep Analysis', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片和数据
        fig.savefig(self.output_dir / 'figure4_power_sweep.png', dpi=300)
        fig.savefig(self.output_dir / 'figure4_power_sweep.pdf')
        df_summary.to_csv(self.output_dir / 'power_sweep_summary.csv', index=False)
        
        return fig, df_summary
    
    def figure5_ocv_soc_relationship(self, battery_params: BatteryParams):
        """
        图5: OCV-SOC关系曲线
        """
        soc_range = np.linspace(0, 1, 101)
        model = BatteryModel(battery_params)
        ocv_values = [model.ocv_func(soc) for soc in soc_range]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(soc_range * 100, ocv_values, 
                color=self.colors['primary'][0], linewidth=2.5)
        
        # 标记关键点
        key_soc_points = [0, 10, 25, 50, 75, 90, 100]
        for soc in key_soc_points:
            ocv = model.ocv_func(soc/100)
            ax.plot(soc, ocv, 'ro', markersize=8)
            ax.annotate(f'({soc}%, {ocv:.3f}V)', 
                       xy=(soc, ocv),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, ha='left')
        
        ax.set_xlabel('State of Charge (%)', fontweight='bold')
        ax.set_ylabel('Open Circuit Voltage (V)', fontweight='bold')
        ax.set_title('OCV-SOC Relationship', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加多项式信息
        coeff_str = "OCV(SOC) = "
        coeffs = battery_params.ocv_coeffs
        for i, coeff in enumerate(coeffs[:3]):  # 只显示前3项
            if i == 0:
                coeff_str += f"{coeff:.3f}"
            else:
                coeff_str += f" + {coeff:.3f}×SOC"
                if i > 1:
                    coeff_str += f"$^{{{i}}}$"
        
        if len(coeffs) > 3:
            coeff_str += " + ..."
        
        ax.text(0.05, 0.95, coeff_str, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图片
        fig.savefig(self.output_dir / 'figure5_ocv_soc_relationship.png', dpi=300)
        fig.savefig(self.output_dir / 'figure5_ocv_soc_relationship.pdf')
        
        # 保存数据
        ocv_data = pd.DataFrame({
            'SOC(%)': soc_range * 100,
            'SOC': soc_range,
            'OCV(V)': ocv_values
        })
        ocv_data.to_csv(self.output_dir / 'ocv_soc_data.csv', index=False)
        
        return fig, ocv_data
    
    def figure6_current_voltage_characteristics(self, battery_params: BatteryParams):
        """
        图6: 电流-电压特性曲线
        """
        power_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
        results_list = []
        
        for P in power_levels:
            solver = BatterySolver(battery_params)
            results = solver.solve_constant_power(P, SolverMethod.RK4, dt=0.5)
            results_list.append((f'{P} W', results))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = plt.cm.plasma(np.linspace(0, 0.8, len(power_levels)))
        
        for idx, (label, results) in enumerate(results_list):
            df = results.to_dataframe()
            color = colors[idx]
            
            # 绘制I-V曲线，用箭头表示时间方向
            ax.plot(df['Voltage(V)'], df['Current(A)'], 
                   color=color, linewidth=2, label=label)
            
            # 添加箭头表示放电方向
            if len(df) > 10:
                mid_idx = len(df) // 4
                start_idx = max(0, mid_idx - 2)
                end_idx = min(len(df) - 1, mid_idx + 2)
                
                ax.annotate('', 
                           xy=(df['Voltage(V)'].iloc[end_idx], df['Current(A)'].iloc[end_idx]),
                           xytext=(df['Voltage(V)'].iloc[start_idx], df['Current(A)'].iloc[start_idx]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        
        ax.set_xlabel('Terminal Voltage (V)', fontweight='bold')
        ax.set_ylabel('Discharge Current (A)', fontweight='bold')
        ax.set_title('Current-Voltage Characteristics', fontsize=13, fontweight='bold')
        ax.legend(title='Power Level', title_fontsize=10, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 添加理论最大功率线
        V_range = np.linspace(battery_params.V_min, battery_params.nominal_voltage, 50)
        for P in [3.0, 4.0, 5.0]:
            I_theoretical = P / V_range
            ax.plot(V_range, I_theoretical, 'k:', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        # 保存图片
        fig.savefig(self.output_dir / 'figure6_iv_characteristics.png', dpi=300)
        fig.savefig(self.output_dir / 'figure6_iv_characteristics.pdf')
        
        return fig, results_list
    
    def figure7_parameter_sensitivity(self, battery_params: BatteryParams):
        """
        图7: 参数敏感性分析
        分析R0、R1、C1、R2、C2对放电性能的影响
        """
        base_params = battery_params
        P0 = 3.0  # 基准功率
        
        # 定义参数变化范围
        param_variations = {
            'R0': [0.025, 0.05, 0.075, 0.10],  # 欧姆内阻
            'R1': [0.01, 0.02, 0.03, 0.04],    # 第一极化电阻
            'C1': [1000, 2000, 3000, 4000],    # 第一极化电容
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for ax_idx, (param_name, param_values) in enumerate(param_variations.items()):
            discharge_times = []
            
            for value in param_values:
                # 创建修改后的参数
                modified_params = BatteryParams(
                    name=f"Modified_{param_name}",
                    Cn=base_params.Cn,
                    SOH=base_params.SOH,
                    R0=value if param_name == 'R0' else base_params.R0,
                    R1=value if param_name == 'R1' else base_params.R1,
                    C1=value if param_name == 'C1' else base_params.C1,
                    R2=base_params.R2,
                    C2=base_params.C2,
                    eta_conv=base_params.eta_conv,
                    SOC0=base_params.SOC0,
                    V_min=base_params.V_min,
                    SOC_min=base_params.SOC_min,
                    ocv_coeffs=base_params.ocv_coeffs.copy()
                )
                
                # 运行仿真
                solver = BatterySolver(modified_params)
                results = solver.solve_constant_power(P0, SolverMethod.RK4, dt=0.5)
                summary = results.get_summary()
                discharge_times.append(summary['termination_time_h'])
            
            # 绘制敏感性分析
            axes[ax_idx].plot(param_values, discharge_times, 'o-', 
                             color=self.colors['primary'][ax_idx], 
                             linewidth=2, markersize=8)
            
            axes[ax_idx].set_xlabel(f'{param_name} Value', fontweight='bold')
            axes[ax_idx].set_ylabel('Discharge Time (h)', fontweight='bold')
            axes[ax_idx].set_title(f'Sensitivity to {param_name}', fontsize=11)
            axes[ax_idx].grid(True, alpha=0.3)
            
            # 标注基准值
            base_value = getattr(base_params, param_name)
            base_time_idx = param_values.index(base_value) if base_value in param_values else -1
            if base_time_idx >= 0:
                axes[ax_idx].plot(base_value, discharge_times[base_time_idx], 'ro', markersize=10)
                axes[ax_idx].annotate('Baseline', 
                                     xy=(base_value, discharge_times[base_time_idx]),
                                     xytext=(5, 5), textcoords='offset points',
                                     fontsize=9, color='red')
        
        fig.suptitle(f'Parameter Sensitivity Analysis (P = {P0} W)', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图片
        fig.savefig(self.output_dir / 'figure7_parameter_sensitivity.png', dpi=300)
        fig.savefig(self.output_dir / 'figure7_parameter_sensitivity.pdf')
        
        return fig
    
    def generate_all_figures(self, battery_params: BatteryParams):
        """
        生成所有图表
        """
        print("Generating all figures for paper...")
        
        # 生成图1: 详细放电曲线
        print("  Generating Figure 1: Detailed discharge curve...")
        fig1, results1 = self.figure1_detailed_discharge(battery_params, P0=2.5)
        plt.close(fig1)
        
        # 生成图2: 电压组成分解
        print("  Generating Figure 2: Voltage composition...")
        fig2 = self.figure2_voltage_composition(battery_params, P0=2.5)
        plt.close(fig2)
        
        # 生成图3: 多功率比较
        print("  Generating Figure 3: Multiple power comparison...")
        fig3, results3 = self.figure3_multiple_power_comparison(battery_params)
        plt.close(fig3)
        
        # 生成图4: 功耗扫描分析
        print("  Generating Figure 4: Power sweep analysis...")
        fig4, summary4 = self.figure4_power_sweep_analysis(battery_params)
        plt.close(fig4)
        
        # 生成图5: OCV-SOC关系
        print("  Generating Figure 5: OCV-SOC relationship...")
        fig5, ocv_data = self.figure5_ocv_soc_relationship(battery_params)
        plt.close(fig5)
        
        # 生成图6: 电流-电压特性
        print("  Generating Figure 6: Current-voltage characteristics...")
        fig6, results6 = self.figure6_current_voltage_characteristics(battery_params)
        plt.close(fig6)
        
        # 生成图7: 参数敏感性分析
        print("  Generating Figure 7: Parameter sensitivity...")
        fig7 = self.figure7_parameter_sensitivity(battery_params)
        plt.close(fig7)
        
        print(f"\nAll figures have been generated and saved in: {self.output_dir}")
        
        # 创建汇总报告
        self.create_summary_report(battery_params, results1, summary4)
        
        return True
    
    def create_summary_report(self, battery_params: BatteryParams, 
                             example_results, power_sweep_summary):
        """创建汇总报告"""
        report_file = self.output_dir / 'simulation_summary.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BATTERY SIMULATION SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. BATTERY PARAMETERS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Name: {battery_params.name}\n")
            f.write(f"  Nominal Voltage: {battery_params.nominal_voltage} V\n")
            f.write(f"  Capacity: {battery_params.Cn} Ah\n")
            f.write(f"  State of Health: {battery_params.SOH*100:.1f}%\n")
            f.write(f"  Ohmic Resistance (R0): {battery_params.R0} Ω\n")
            f.write(f"  Polarization R1: {battery_params.R1} Ω, C1: {battery_params.C1} F\n")
            f.write(f"  Polarization R2: {battery_params.R2} Ω, C2: {battery_params.C2} F\n")
            f.write(f"  Time Constants: τ₁={battery_params.tau1:.0f} s, τ₂={battery_params.tau2:.0f} s\n")
            f.write(f"  Coulombic Efficiency: {battery_params.eta*100:.1f}%\n")
            f.write(f"  Power Conversion Efficiency: {battery_params.eta_conv*100:.1f}%\n")
            
            f.write("\n2. EXAMPLE SIMULATION (2.5 W)\n")
            f.write("-" * 40 + "\n")
            if example_results:
                summary = example_results.get_summary()
                f.write(f"  Termination Reason: {summary['termination_reason']}\n")
                f.write(f"  Discharge Time: {summary['termination_time_h']:.2f} h\n")
                f.write(f"  Final SOC: {summary['final_SOC']*100:.1f}%\n")
                f.write(f"  Final Voltage: {summary['final_voltage']:.3f} V\n")
                f.write(f"  Total Capacity: {summary['total_capacity_Ah']:.3f} Ah\n")
                f.write(f"  Total Energy: {summary['total_energy_Wh']:.3f} Wh\n")
                f.write(f"  Current Range: {summary['min_current']:.3f} - {summary['max_current']:.3f} A\n")
                f.write(f"  Average Power: {summary['average_power']:.3f} W\n")
            
            f.write("\n3. POWER SWEEP SUMMARY\n")
            f.write("-" * 40 + "\n")
            if isinstance(power_sweep_summary, pd.DataFrame) and not power_sweep_summary.empty:
                f.write("  Power(W)  Time(h)   Capacity(Ah)  Energy(Wh)\n")
                f.write("  " + "-"*45 + "\n")
                for _, row in power_sweep_summary.iterrows():
                    f.write(f"  {row['Power(W)']:7.1f}  {row['termination_time_h']:8.2f}  "
                           f"{row['total_capacity_Ah']:12.3f}  {row['total_energy_Wh']:10.2f}\n")
            
            f.write("\n4. GENERATED FIGURES\n")
            f.write("-" * 40 + "\n")
            figures = [
                "Figure 1: Detailed discharge characteristics",
                "Figure 2: Voltage composition analysis", 
                "Figure 3: Multiple power level comparison",
                "Figure 4: Power sweep analysis",
                "Figure 5: OCV-SOC relationship",
                "Figure 6: Current-voltage characteristics",
                "Figure 7: Parameter sensitivity analysis"
            ]
            
            for fig in figures:
                f.write(f"  • {fig}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")
        
        print(f"Summary report saved to: {report_file}")

# =============================================
# 5. 主程序
# =============================================

def main():
    """主程序"""
    print("=" * 60)
    print("BATTERY SIMULATION FOR PAPER FIGURES")
    print("=" * 60)
    
    # 1. 创建标准电池参数
    battery_params = BatteryParams(
        name="Li-ion_3Ah",
        Cn=3.0,  # 3Ah容量
        SOH=0.95,
        R0=0.05,
        R1=0.02, C1=2000,
        R2=0.03, C2=10000,
        nominal_voltage=3.7,
        eta_conv=0.95,
        SOC0=1.0,
        V_min=3.2,
        SOC_min=0.05,
        t_max=36000,
        ocv_coeffs=[3.259, 6.069, -36.387, 132.659, -291.517, 
                   371.911, -248.739, 66.939]
    )
    
    # 2. 创建图表生成器
    figure_generator = BatteryPaperFigures()
    
    # 3. 生成所有图表
    figure_generator.generate_all_figures(battery_params)
    
    # 4. 显示完成信息
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"All figures have been generated and saved in: {figure_generator.output_dir}")
    print("\nGenerated files:")
    for file in sorted(figure_generator.output_dir.glob("*")):
        size_kb = file.stat().st_size / 1024
        print(f"  • {file.name} ({size_kb:.1f} KB)")
    
    print("\nYou can now use these figures in your paper.")

# =============================================
# 6. 程序入口
# =============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()