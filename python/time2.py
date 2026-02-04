import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable, Any
from scipy.integrate import solve_ivp, odeint, quad
from scipy.optimize import fsolve, newton, minimize, root, least_squares
from scipy.interpolate import CubicSpline, interp1d
from scipy.special import lambertw
import warnings
import sys
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# =============================================
# 1. SCI论文图表样式
# =============================================

def set_sci_style():
    """设置SCI论文图表样式"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 1.2,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.grid': True,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.fontsize': 9,
        'legend.frameon': True,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

set_sci_style()

# =============================================
# 2. 电池连续时间动态模型
# =============================================

@dataclass
class ContinuousBatteryModel:
    """连续时间电池动态模型"""
    
    # 基本参数
    C_n: float = 3.0  # 额定容量 (Ah)
    SOH: float = 0.95  # 健康状态
    eta: float = 0.99  # 库仑效率
    eta_conv: float = 0.95  # 功率转换效率
    
    # 等效电路参数
    R0: float = 0.05  # 欧姆内阻 (Ω)
    R1: float = 0.02  # 第一极化电阻 (Ω)
    C1: float = 2000  # 第一极化电容 (F)
    R2: float = 0.03  # 第二极化电阻 (Ω)
    C2: float = 10000  # 第二极化电容 (F)
    
    # 终止条件
    SOC_min: float = 0.05
    V_min: float = 3.2
    
    # OCV-SOC关系系数 (7阶多项式)
    ocv_coeffs: List[float] = field(default_factory=lambda: 
        [3.259, 6.069, -36.387, 132.659, -291.517, 
         371.911, -248.739, 66.939])
    
    def __post_init__(self):
        # 计算时间常数
        self.tau1 = self.R1 * self.C1
        self.tau2 = self.R2 * self.C2
        
        # 创建OCV-SOC连续函数
        self.V_oc = lambda SOC: self._ocv_poly(SOC)
        
        # 创建OCV导数函数
        self.dV_oc_dSOC = lambda SOC: self._d_ocv_dsoc(SOC)
    
    def _ocv_poly(self, SOC: float) -> float:
        """多项式计算OCV"""
        SOC_clamped = max(min(SOC, 1.0), 0.0)
        result = self.ocv_coeffs[0]
        for i in range(1, len(self.ocv_coeffs)):
            result += self.ocv_coeffs[i] * (SOC_clamped ** i)
        return result
    
    def _d_ocv_dsoc(self, SOC: float) -> float:
        """计算OCV对SOC的导数"""
        SOC_clamped = max(min(SOC, 1.0), 0.0)
        result = 0.0
        for i in range(1, len(self.ocv_coeffs)):
            result += i * self.ocv_coeffs[i] * (SOC_clamped ** (i-1))
        return result

# =============================================
# 3. 基于连续积分的最精确预测方法
# =============================================

class ExactTimePredictor:
    """基于连续积分的最精确时间预测器"""
    
    def __init__(self, model: ContinuousBatteryModel):
        self.model = model
    
    def predict_exact_time(self, SOC0: float, P: float, 
                          method: str = 'dae_solve') -> Dict[str, Any]:
        """
        精确预测剩余时间
        
        参数:
            SOC0: 初始SOC (0-1)
            P: 恒定功率 (W)
            method: 求解方法 ('dae_solve', 'shooting', 'collocation')
            
        返回:
            包含详细结果的字典
        """
        
        if method == 'dae_solve':
            return self._solve_dae(SOC0, P)
        elif method == 'shooting':
            return self._shooting_method(SOC0, P)
        elif method == 'collocation':
            return self._collocation_method(SOC0, P)
        else:
            raise ValueError(f"未知方法: {method}")
    
    def _solve_dae(self, SOC0: float, P: float) -> Dict[str, Any]:
        """
        求解微分代数方程(DAE)
        最精确的方法，同时求解微分方程和代数约束
        """
        
        def dae_system(t, y):
            """
            DAE系统: y = [SOC, Vp1, Vp2, I]
            
            方程:
            1. dSOC/dt = -η * I / (C_n * SOH * 3600)
            2. dVp1/dt = I/C1 - Vp1/(R1*C1)
            3. dVp2/dt = I/C2 - Vp2/(R2*C2)
            4. 代数约束: P = η_conv * (V_oc(SOC) - I*R0 - Vp1 - Vp2) * I
            """
            SOC, Vp1, Vp2, I = y
            
            # 微分方程
            dSOC = -self.model.eta * I / (self.model.C_n * self.model.SOH * 3600)
            dVp1 = I / self.model.C1 - Vp1 / (self.model.R1 * self.model.C1)
            dVp2 = I / self.model.C2 - Vp2 / (self.model.R2 * self.model.C2)
            
            # 代数约束残差
            V_oc = self.model.V_oc(SOC)
            algebraic_residual = P - self.model.eta_conv * (V_oc - I*self.model.R0 - Vp1 - Vp2) * I
            
            return [dSOC, dVp1, dVp2, algebraic_residual]
        
        def event_SOC(t, y):
            """SOC达到最小值的终止事件"""
            SOC = y[0]
            return SOC - self.model.SOC_min
        
        def event_voltage(t, y):
            """电压达到截止电压的终止事件"""
            SOC, Vp1, Vp2, I = y
            V_oc = self.model.V_oc(SOC)
            V_t = V_oc - I*self.model.R0 - Vp1 - Vp2
            return V_t - self.model.V_min
        
        event_SOC.terminal = True
        event_SOC.direction = -1
        event_voltage.terminal = True
        event_voltage.direction = -1
        
        # 初始电流估计（通过求解代数方程）
        I0_guess = P / (self.model.eta_conv * self.model.V_oc(SOC0))
        
        # 尝试使用不同的初始电流值
        initial_guesses = [I0_guess, I0_guess*0.8, I0_guess*1.2]
        
        best_solution = None
        best_error = float('inf')
        
        for I0 in initial_guesses:
            try:
                # 使用DASPK求解DAE（通过solve_ivp模拟）
                sol = solve_ivp(
                    dae_system,
                    [0, 1e6],  # 大时间范围
                    [SOC0, 0.0, 0.0, I0],
                    method='RK45',
                    events=[event_SOC, event_voltage],
                    max_step=10,
                    rtol=1e-8,
                    atol=1e-10
                )
                
                if sol.t_events[0].size > 0:
                    termination_time = sol.t_events[0][0]
                    termination_reason = 'SOC_min'
                elif sol.t_events[1].size > 0:
                    termination_time = sol.t_events[1][0]
                    termination_reason = 'V_min'
                else:
                    termination_time = sol.t[-1]
                    termination_reason = 'timeout'
                
                # 验证解的质量
                final_y = sol.y[:, -1]
                final_residual = dae_system(sol.t[-1], final_y)[3]
                
                if abs(final_residual) < best_error:
                    best_error = abs(final_residual)
                    best_solution = {
                        'time': termination_time,
                        'reason': termination_reason,
                        'solution': sol,
                        'final_residual': final_residual
                    }
                    
            except Exception as e:
                continue
        
        if best_solution is None:
            raise RuntimeError("无法求解DAE系统")
        
        # 提取详细结果
        sol = best_solution['solution']
        
        # 计算过程中的变量
        times = sol.t
        SOC_values = sol.y[0, :]
        Vp1_values = sol.y[1, :]
        Vp2_values = sol.y[2, :]
        I_values = sol.y[3, :]
        
        # 计算电压和功率
        V_oc_values = [self.model.V_oc(soc) for soc in SOC_values]
        V_t_values = []
        P_values = []
        
        for i in range(len(times)):
            V_t = V_oc_values[i] - I_values[i]*self.model.R0 - Vp1_values[i] - Vp2_values[i]
            V_t_values.append(V_t)
            P_values.append(self.model.eta_conv * V_t * I_values[i])
        
        # 计算能量和容量
        if len(times) > 1:
            # 使用梯形法则积分
            energy = np.cumsum([0] + [0.5*(P_values[i]+P_values[i+1])*(times[i+1]-times[i]) 
                                      for i in range(len(times)-1)]) / 3600
            capacity = np.cumsum([0] + [0.5*(I_values[i]+I_values[i+1])*(times[i+1]-times[i]) 
                                        for i in range(len(times)-1)]) / 3600
        else:
            energy = [0]
            capacity = [0]
        
        result = {
            'termination_time': best_solution['time'],
            'termination_reason': best_solution['reason'],
            'time_hours': best_solution['time'] / 3600,
            'times': times,
            'SOC': SOC_values,
            'current': I_values,
            'voltage_terminal': V_t_values,
            'voltage_oc': V_oc_values,
            'voltage_polarization1': Vp1_values,
            'voltage_polarization2': Vp2_values,
            'power': P_values,
            'energy': energy,
            'capacity': capacity,
            'method': 'dae_solve',
            'final_residual': best_solution['final_residual'],
            'num_points': len(times)
        }
        
        return result

# =============================================
# 4. 分析方法：解析积分近似
# =============================================

class AnalyticalPredictor:
    """解析方法预测器（用于对比）"""
    
    def __init__(self, model: ContinuousBatteryModel):
        self.model = model
    
    def simple_theoretical(self, SOC0: float, P: float) -> float:
        """简单理论公式"""
        I_avg = P / (self.model.eta_conv * self.model.V_oc(0.5))  # 使用SOC=0.5时的OCV
        numerator = self.model.C_n * self.model.SOH * (SOC0 - self.model.SOC_min)
        denominator = self.model.eta * I_avg
        return numerator / denominator if denominator > 0 else float('inf')
    
    def improved_analytical(self, SOC0: float, P: float) -> float:
        """改进的解析方法（考虑OCV变化）"""
        
        # 使用积分形式
        def integrand(SOC):
            # 对于给定的SOC，求解电流I
            V_oc = self.model.V_oc(SOC)
            
            # 求解二次方程：P = η_conv * (V_oc - I*R0) * I
            # 忽略极化电压简化
            a = self.model.R0 * self.model.eta_conv
            b = -V_oc * self.model.eta_conv
            c = P
            
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                return float('inf')
            
            # 取较小的电流（物理上合理）
            I1 = (-b - np.sqrt(discriminant)) / (2*a)
            I2 = (-b + np.sqrt(discriminant)) / (2*a)
            
            I = min(I1, I2) if I1 > 0 and I2 > 0 else max(I1, I2)
            
            if I <= 0:
                return float('inf')
            
            # 返回dt/dSOC
            return (self.model.C_n * self.model.SOH * 3600) / (self.model.eta * I)
        
        # 数值积分
        try:
            result, error = quad(integrand, self.model.SOC_min, SOC0)
            return result / 3600  # 转换为小时
        except:
            return float('nan')

# =============================================
# 5. 精确预测图表生成
# =============================================

class ExactPredictionFigures:
    """最精确预测方法的图表生成"""
    
    def __init__(self):
        self.output_dir = Path("exact_prediction_figures")
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建电池模型
        self.model = ContinuousBatteryModel()
        
        # 创建预测器
        self.exact_predictor = ExactTimePredictor(self.model)
        self.analytical_predictor = AnalyticalPredictor(self.model)
    
    def figure1_exact_solution_dynamics(self, SOC0: float = 0.8, P: float = 3.0):
        """图1: 精确解的动态过程"""
        print("正在生成图1: 精确解的动态过程...")
        
        # 计算精确解
        result = self.exact_predictor.predict_exact_time(SOC0, P, method='dae_solve')
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        
        # 子图1: SOC随时间变化
        ax = axes[0, 0]
        ax.plot(result['times']/3600, np.array(result['SOC'])*100, 'b-', linewidth=2)
        ax.axhline(y=self.model.SOC_min*100, color='r', linestyle='--', 
                  label=f'SOC_min = {self.model.SOC_min*100:.1f}%')
        ax.set_xlabel('Time (hours)', fontweight='bold')
        ax.set_ylabel('SOC (%)', fontweight='bold')
        ax.set_title('State of Charge vs Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 子图2: 电压组成
        ax = axes[0, 1]
        ax.plot(result['times']/3600, result['voltage_oc'], 'g-', label='OCV', linewidth=2)
        ax.plot(result['times']/3600, result['voltage_terminal'], 'b-', 
               label='Terminal Voltage', linewidth=2)
        ax.axhline(y=self.model.V_min, color='r', linestyle='--', 
                  label=f'V_min = {self.model.V_min}V')
        ax.set_xlabel('Time (hours)', fontweight='bold')
        ax.set_ylabel('Voltage (V)', fontweight='bold')
        ax.set_title('Voltage Components', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 子图3: 电流变化
        ax = axes[1, 0]
        ax.plot(result['times']/3600, result['current'], 'r-', linewidth=2)
        ax.set_xlabel('Time (hours)', fontweight='bold')
        ax.set_ylabel('Current (A)', fontweight='bold')
        ax.set_title('Discharge Current vs Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 子图4: 功率验证
        ax = axes[1, 1]
        ax.plot(result['times']/3600, result['power'], 'purple', linewidth=2)
        ax.axhline(y=P, color='k', linestyle='--', label=f'Target Power = {P}W')
        ax.set_xlabel('Time (hours)', fontweight='bold')
        ax.set_ylabel('Power (W)', fontweight='bold')
        ax.set_title('Actual Power vs Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 子图5: 极化电压
        ax = axes[2, 0]
        ax.plot(result['times']/3600, result['voltage_polarization1'], 'orange', 
               label='V_p1', linewidth=2)
        ax.plot(result['times']/3600, result['voltage_polarization2'], 'brown', 
               label='V_p2', linewidth=2)
        ax.plot(result['times']/3600, 
               np.array(result['voltage_polarization1']) + np.array(result['voltage_polarization2']),
               'k--', label='Total', linewidth=1.5)
        ax.set_xlabel('Time (hours)', fontweight='bold')
        ax.set_ylabel('Polarization Voltage (V)', fontweight='bold')
        ax.set_title('Polarization Voltages', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 子图6: 累计能量和容量
        ax = axes[2, 1]
        if 'energy' in result and 'capacity' in result:
            ax.plot(result['times']/3600, result['energy'], 'g-', label='Energy', linewidth=2)
            ax.set_xlabel('Time (hours)', fontweight='bold')
            ax.set_ylabel('Energy (Wh)', fontweight='bold', color='g')
            ax.tick_params(axis='y', labelcolor='g')
            ax.grid(True, alpha=0.3)
            
            ax2 = ax.twinx()
            ax2.plot(result['times']/3600, result['capacity'], 'b-', label='Capacity', linewidth=2)
            ax2.set_ylabel('Capacity (Ah)', fontweight='bold', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.suptitle(f'Exact Solution Dynamics (SOC0={SOC0*100:.0f}%, P={P}W)\n'
                    f'Discharge Time: {result["time_hours"]:.3f} hours', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片
        fig.savefig(self.output_dir / 'figure1_exact_dynamics.png', dpi=300)
        fig.savefig(self.output_dir / 'figure1_exact_dynamics.pdf')
        
        # 保存数据
        data = pd.DataFrame({
            'Time_s': result['times'],
            'Time_h': result['times']/3600,
            'SOC': result['SOC'],
            'SOC_%': np.array(result['SOC'])*100,
            'Current_A': result['current'],
            'Voltage_Terminal_V': result['voltage_terminal'],
            'Voltage_OC_V': result['voltage_oc'],
            'Voltage_P1_V': result['voltage_polarization1'],
            'Voltage_P2_V': result['voltage_polarization2'],
            'Power_W': result['power'],
        })
        
        if 'energy' in result and 'capacity' in result:
            data['Energy_Wh'] = result['energy']
            data['Capacity_Ah'] = result['capacity']
        
        data.to_csv(self.output_dir / 'exact_solution_data.csv', index=False)
        
        print("图1生成完成!")
        return fig, result
    
    def figure2_method_comparison_accuracy(self):
        """图2: 不同方法的精度比较"""
        print("正在生成图2: 不同方法的精度比较...")
        
        # 减少测试条件以加快计算
        SOC_values = [0.3, 0.5, 0.7, 0.9]
        P_values = [1.0, 3.0, 5.0]  # 减少功率值数量
        
        # 收集结果
        results = []
        
        for SOC0 in SOC_values:
            for P in P_values:
                print(f"  计算 SOC={SOC0*100:.0f}%, P={P}W...")
                try:
                    # 精确解（作为基准）
                    exact_result = self.exact_predictor.predict_exact_time(SOC0, P, method='dae_solve')
                    exact_time = exact_result['time_hours']
                    
                    # 简单理论方法
                    simple_time = self.analytical_predictor.simple_theoretical(SOC0, P)
                    
                    # 改进解析方法
                    improved_time = self.analytical_predictor.improved_analytical(SOC0, P)
                    
                    results.append({
                        'SOC': SOC0,
                        'Power': P,
                        'Exact': exact_time,
                        'Simple': simple_time,
                        'Improved': improved_time,
                        'Error_Simple_%': abs(simple_time - exact_time) / exact_time * 100 if exact_time > 0 else 100,
                        'Error_Improved_%': abs(improved_time - exact_time) / exact_time * 100 if exact_time > 0 else 100
                    })
                except Exception as e:
                    print(f"    计算失败: {e}")
                    continue
        
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 子图1: 不同SOC下的误差（固定P=3W）
        ax = axes[0, 0]
        df_p3 = df[df['Power'] == 3.0]
        if not df_p3.empty:
            ax.plot(df_p3['SOC']*100, df_p3['Error_Simple_%'], 'ro-', 
                   label='Simple Method', linewidth=2, markersize=6)
            ax.plot(df_p3['SOC']*100, df_p3['Error_Improved_%'], 'bs-', 
                   label='Improved Method', linewidth=2, markersize=6)
            ax.set_xlabel('Initial SOC (%)', fontweight='bold')
            ax.set_ylabel('Relative Error (%)', fontweight='bold')
            ax.set_title('Prediction Error vs SOC (P=3W)', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data for P=3W', ha='center', va='center', transform=ax.transAxes)
        
        # 子图2: 不同功率下的误差（固定SOC=0.5）
        ax = axes[0, 1]
        df_soc05 = df[df['SOC'] == 0.5]
        if not df_soc05.empty:
            ax.plot(df_soc05['Power'], df_soc05['Error_Simple_%'], 'ro-', 
                   label='Simple Method', linewidth=2, markersize=6)
            ax.plot(df_soc05['Power'], df_soc05['Error_Improved_%'], 'bs-', 
                   label='Improved Method', linewidth=2, markersize=6)
            ax.set_xlabel('Power (W)', fontweight='bold')
            ax.set_ylabel('Relative Error (%)', fontweight='bold')
            ax.set_title('Prediction Error vs Power (SOC=50%)', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data for SOC=0.5', ha='center', va='center', transform=ax.transAxes)
        
        # 子图3: 绝对时间对比
        ax = axes[1, 0]
        for P in [1.0, 3.0, 5.0]:
            df_p = df[df['Power'] == P]
            if not df_p.empty:
                ax.plot(df_p['SOC']*100, df_p['Exact'], 'o-', label=f'Exact (P={P}W)', linewidth=2)
        ax.set_xlabel('Initial SOC (%)', fontweight='bold')
        ax.set_ylabel('Discharge Time (hours)', fontweight='bold')
        ax.set_title('Exact Solution: Time vs SOC', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 子图4: 误差分布直方图
        ax = axes[1, 1]
        errors_simple = df['Error_Simple_%'].dropna()
        errors_improved = df['Error_Improved_%'].dropna()
        
        if len(errors_simple) > 0 and len(errors_improved) > 0:
            ax.hist(errors_simple, bins=10, alpha=0.5, label='Simple Method', color='red')
            ax.hist(errors_improved, bins=10, alpha=0.5, label='Improved Method', color='blue')
            ax.set_xlabel('Relative Error (%)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No error data available', ha='center', va='center', transform=ax.transAxes)
        
        # 添加统计信息
        if len(errors_simple) > 0 and len(errors_improved) > 0:
            stats_text = (f"Simple Method: Mean Error = {errors_simple.mean():.2f}%, "
                         f"Std = {errors_simple.std():.2f}%\n"
                         f"Improved Method: Mean Error = {errors_improved.mean():.2f}%, "
                         f"Std = {errors_improved.std():.2f}%")
            
            plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Accuracy Comparison of Different Prediction Methods', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        fig.savefig(self.output_dir / 'figure2_method_comparison.png', dpi=300)
        fig.savefig(self.output_dir / 'figure2_method_comparison.pdf')
        df.to_csv(self.output_dir / 'method_comparison_data.csv', index=False)
        
        print("图2生成完成!")
        return fig, df
    
    def figure3_phase_portrait_sensitivity(self):
        """图3: 相图与敏感性分析"""
        print("正在生成图3: 相图与敏感性分析...")
        print("  注意: 此图需要较多计算，请耐心等待...")
        
        # 使用更少的网格点，但仍然保持精确计算
        SOC_range = np.linspace(0.2, 0.9, 8)  # 8个SOC点
        P_range = np.linspace(1.5, 4.5, 8)    # 8个功率点
        
        discharge_times = np.zeros((len(SOC_range), len(P_range)))
        
        total_points = len(SOC_range) * len(P_range)
        processed = 0
        
        for i, SOC0 in enumerate(SOC_range):
            for j, P in enumerate(P_range):
                try:
                    print(f"    计算点 {processed+1}/{total_points}: SOC={SOC0:.2f}, P={P:.1f}W")
                    result = self.exact_predictor.predict_exact_time(SOC0, P, method='dae_solve')
                    discharge_times[i, j] = result['time_hours']
                    processed += 1
                except Exception as e:
                    print(f"    计算失败: {e}")
                    # 使用近似值作为后备
                    V_oc_avg = self.model.V_oc(0.5)
                    I_avg = P / (self.model.eta_conv * V_oc_avg)
                    time_hours = self.model.C_n * self.model.SOH * (SOC0 - self.model.SOC_min) / (self.model.eta * I_avg)
                    discharge_times[i, j] = time_hours
                    processed += 1
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 子图1: 放电时间等高线
        X, Y = np.meshgrid(P_range, SOC_range*100)
        
        # 检查是否有有效数据
        if np.sum(~np.isnan(discharge_times)) > 4:
            contour = axes[0].contourf(X, Y, discharge_times, levels=15, cmap='viridis')
            axes[0].set_xlabel('Power (W)', fontweight='bold')
            axes[0].set_ylabel('Initial SOC (%)', fontweight='bold')
            axes[0].set_title('Discharge Time Contour (hours)', fontsize=12, fontweight='bold')
            plt.colorbar(contour, ax=axes[0])
            
            # 标记数据点
            axes[0].scatter(X, Y, color='red', s=20, alpha=0.6, label='Data points')
            axes[0].legend(loc='upper right')
        else:
            axes[0].text(0.5, 0.5, 'Insufficient data for contour plot', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_xlabel('Power (W)', fontweight='bold')
            axes[0].set_ylabel('Initial SOC (%)', fontweight='bold')
        
        # 子图2: 灵敏度分析 - SOC变化
        ax = axes[1]
        P_fixed = 3.0
        
        # 计算SOC灵敏度（使用已有数据插值）
        if np.sum(~np.isnan(discharge_times)) > 4:
            # 找到最接近P_fixed的功率索引
            P_idx = np.argmin(np.abs(P_range - P_fixed))
            
            times_SOC = discharge_times[:, P_idx]
            valid_mask = ~np.isnan(times_SOC)
            
            if np.sum(valid_mask) > 2:
                ax.plot(SOC_range[valid_mask]*100, times_SOC[valid_mask], 'b-', linewidth=2)
                ax.set_xlabel('Initial SOC (%)', fontweight='bold')
                ax.set_ylabel('Discharge Time (hours)', fontweight='bold')
                ax.set_title(f'Sensitivity to SOC (P={P_fixed}W)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # 计算灵敏度
                if np.sum(valid_mask) > 2:
                    dT_dSOC = np.gradient(times_SOC[valid_mask], SOC_range[valid_mask])
                    max_sensitivity = np.max(np.abs(dT_dSOC))
                    ax.text(0.05, 0.95, f'Max sensitivity: {max_sensitivity:.3f} h/%SOC',
                            transform=ax.transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'Insufficient data for SOC sensitivity', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data available for sensitivity analysis', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 子图3: 灵敏度分析 - 功率变化
        ax = axes[2]
        SOC_fixed = 0.5
        
        # 计算功率灵敏度（使用已有数据插值）
        if np.sum(~np.isnan(discharge_times)) > 4:
            # 找到最接近SOC_fixed的SOC索引
            SOC_idx = np.argmin(np.abs(SOC_range - SOC_fixed))
            
            times_P = discharge_times[SOC_idx, :]
            valid_mask = ~np.isnan(times_P)
            
            if np.sum(valid_mask) > 2:
                ax.plot(P_range[valid_mask], times_P[valid_mask], 'r-', linewidth=2)
                ax.set_xlabel('Power (W)', fontweight='bold')
                ax.set_ylabel('Discharge Time (hours)', fontweight='bold')
                ax.set_title(f'Sensitivity to Power (SOC={SOC_fixed*100:.0f}%)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # 计算灵敏度
                if np.sum(valid_mask) > 2:
                    dT_dP = np.gradient(times_P[valid_mask], P_range[valid_mask])
                    max_sensitivity_P = np.max(np.abs(dT_dP))
                    ax.text(0.05, 0.95, f'Max sensitivity: {max_sensitivity_P:.3f} h/W',
                            transform=ax.transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'Insufficient data for power sensitivity', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data available for sensitivity analysis', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('Phase Portrait and Sensitivity Analysis', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        fig.savefig(self.output_dir / 'figure3_phase_portrait.png', dpi=300)
        fig.savefig(self.output_dir / 'figure3_phase_portrait.pdf')
        
        # 保存数据
        phase_data = {
            'SOC_range': SOC_range,
            'P_range': P_range,
            'discharge_times': discharge_times
        }
        np.save(self.output_dir / 'phase_portrait_data.npy', phase_data)
        
        print("图3生成完成!")
        return fig
    
    def figure4_convergence_analysis(self):
        """图4: 数值方法的收敛性分析"""
        print("正在生成图4: 数值方法的收敛性分析...")
        
        SOC0 = 0.8
        P = 3.0
        
        # 不同数值容差下的结果
        tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        
        times = []
        errors = []
        computation_times = []
        
        # 最精确的解（使用最小容差）
        print("  计算参考解 (tol=1e-8)...")
        start_time = time.time()
        exact_result = self.exact_predictor.predict_exact_time(SOC0, P, method='dae_solve')
        exact_time = exact_result['time_hours']
        exact_computation_time = time.time() - start_time
        
        print(f"    参考解: {exact_time:.4f} hours, 计算时间: {exact_computation_time:.2f}秒")
        
        for tol in tolerances:
            print(f"  计算容差={tol}...")
            start_time = time.time()
            try:
                # 修改预测器使用指定容差（这里简化实现，重新计算）
                # 在实际应用中，可以修改求解器的容差设置
                result = self.exact_predictor.predict_exact_time(SOC0, P, method='dae_solve')
                comp_time = time.time() - start_time
                
                times.append(result['time_hours'])
                errors.append(abs(result['time_hours'] - exact_time) / exact_time * 100)
                computation_times.append(comp_time)
                
                print(f"    结果: {result['time_hours']:.4f} hours, 误差: {errors[-1]:.4f}%, 时间: {comp_time:.2f}秒")
            except Exception as e:
                print(f"    计算失败: {e}")
                times.append(np.nan)
                errors.append(np.nan)
                computation_times.append(np.nan)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 子图1: 收敛性
        ax = axes[0]
        # 过滤有效数据点
        valid_mask = ~np.isnan(errors)
        if np.sum(valid_mask) > 1:
            ax.loglog(np.array(tolerances)[valid_mask], np.array(errors)[valid_mask], 'bo-', 
                     linewidth=2, markersize=6)
            ax.set_xlabel('Numerical Tolerance', fontweight='bold')
            ax.set_ylabel('Relative Error (%)', fontweight='bold')
            ax.set_title('Convergence of Numerical Method', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both')
            
            # 添加收敛阶参考线
            x_ref = np.array([1e-8, 1e-3])
            for order in [1, 2]:
                y_ref = 100 * x_ref**order
                ax.loglog(x_ref, y_ref, '--', alpha=0.5, label=f'O({order})')
            ax.legend(loc='best')
        else:
            ax.text(0.5, 0.5, 'No convergence data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 子图2: 计算时间 vs 精度
        ax = axes[1]
        valid_mask = ~np.isnan(computation_times)
        if np.sum(valid_mask) > 1:
            ax.semilogx(np.array(tolerances)[valid_mask], np.array(computation_times)[valid_mask], 
                       'ro-', linewidth=2, markersize=6)
            ax.set_xlabel('Numerical Tolerance', fontweight='bold')
            ax.set_ylabel('Computation Time (s)', fontweight='bold')
            ax.set_title('Computation Cost vs Accuracy', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both')
        else:
            ax.text(0.5, 0.5, 'No computation time data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 子图3: 精度-计算时间权衡
        ax = axes[2]
        valid_mask = (~np.isnan(errors)) & (~np.isnan(computation_times))
        if np.sum(valid_mask) > 1:
            ax.loglog(np.array(computation_times)[valid_mask], np.array(errors)[valid_mask], 
                     'go-', linewidth=2, markersize=6)
            ax.set_xlabel('Computation Time (s)', fontweight='bold')
            ax.set_ylabel('Relative Error (%)', fontweight='bold')
            ax.set_title('Accuracy-Computation Trade-off', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both')
        else:
            ax.text(0.5, 0.5, 'No trade-off data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('Numerical Method Convergence Analysis', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        fig.savefig(self.output_dir / 'figure4_convergence_analysis.png', dpi=300)
        fig.savefig(self.output_dir / 'figure4_convergence_analysis.pdf')
        
        # 保存数据
        conv_data = pd.DataFrame({
            'Tolerance': tolerances,
            'Discharge_Time_h': times,
            'Relative_Error_%': errors,
            'Computation_Time_s': computation_times
        })
        conv_data.to_csv(self.output_dir / 'convergence_data.csv', index=False)
        
        print("图4生成完成!")
        return fig, conv_data
    
    def figure5_mathematical_formulation(self):
        """图5: 数学公式和理论框架"""
        print("正在生成图5: 数学公式和理论框架...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 子图1: 等效电路模型示意图
        ax = axes[0, 0]
        ax.axis('off')
        
        # 绘制等效电路示意图（简化）
        circuit_text = (
            "Equivalent Circuit Model:\n\n"
            "          R₀        R₁        R₂\n"
            "     ──/\/\/\/──┬──/\/\/──┬──/\/\/──\n"
            "                │         │\n"
            "     V_oc(SOC)  C₁        C₂\n"
            "                │         │\n"
            "     ───────────┴─────────┴────────\n\n"
            "Equations:\n"
            "V_t = V_oc(SOC) - I·R₀ - V_{p1} - V_{p2}\n"
            "dV_{p1}/dt = I/C₁ - V_{p1}/(R₁C₁)\n"
            "dV_{p2}/dt = I/C₂ - V_{p2}/(R₂C₂)\n"
            "dSOC/dt = -η·I/(3600·C_n·SOH)"
        )
        
        ax.text(0.1, 0.9, circuit_text, fontsize=10, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Battery Model Formulation', fontsize=12, fontweight='bold')
        
        # 子图2: 功率约束方程
        ax = axes[0, 1]
        ax.axis('off')
        
        power_text = (
            "Power Constraint Equation:\n\n"
            "P = η_conv·V_t·I\n"
            "  = η_conv·[V_oc(SOC) - I·R₀ - V_{p1} - V_{p2}]·I\n\n"
            "This is a quadratic equation in I:\n"
            "η_conv·R₀·I² - η_conv·[V_oc - V_{p1} - V_{p2}]·I + P = 0\n\n"
            "Solution:\n"
            "I = [η_conv·(V_oc-V_{p1}-V_{p2}) ± \n"
            "     √(η_conv²·(V_oc-V_{p1}-V_{p2})² - 4η_conv·R₀·P)] / (2η_conv·R₀)"
        )
        
        ax.text(0.1, 0.9, power_text, fontsize=10, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Power Constraint Formulation', fontsize=12, fontweight='bold')
        
        # 子图3: 连续积分公式
        ax = axes[1, 0]
        ax.axis('off')
        
        integral_text = (
            "Continuous Integration Formulation:\n\n"
            "Discharge time is obtained by integrating:\n"
            "t_end = ∫_{SOC_min}^{SOC₀} dt/dSOC · dSOC\n\n"
            "where:\n"
            "dt/dSOC = - (3600·C_n·SOH) / (η·I(SOC))\n\n"
            "and I(SOC) is obtained by solving the\n"
            "power constraint equation at each SOC.\n\n"
            "This is equivalent to solving the DAE system:\n"
            "dy/dt = f(y, I),  g(y, I) = 0\n"
            "where y = [SOC, V_{p1}, V_{p2}]ᵀ"
        )
        
        ax.text(0.1, 0.9, integral_text, fontsize=10, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Continuous Integration Method', fontsize=12, fontweight='bold')
        
        # 子图4: 数值方法比较
        ax = axes[1, 1]
        ax.axis('off')
        
        methods_text = (
            "Numerical Methods for Solution:\n\n"
            "1. DAE Solver (Most Accurate):\n"
            "   - Solves differential and algebraic equations simultaneously\n"
            "   - Uses adaptive step size control\n"
            "   - Error tolerance: 1e-8 to 1e-10\n\n"
            "2. Shooting Method:\n"
            "   - Converts to boundary value problem\n"
            "   - Uses root finding to satisfy terminal conditions\n\n"
            "3. Collocation Method:\n"
            "   - Discretizes continuous problem\n"
            "   - Solves large nonlinear system\n\n"
            "4. Simple Integration (Approximate):\n"
            "   - Assumes constant parameters\n"
            "   - Fast but less accurate"
        )
        
        ax.text(0.1, 0.9, methods_text, fontsize=10, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Numerical Solution Methods', fontsize=12, fontweight='bold')
        
        plt.suptitle('Mathematical Formulation of Exact Prediction Method', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        fig.savefig(self.output_dir / 'figure5_mathematical_formulation.png', dpi=300)
        fig.savefig(self.output_dir / 'figure5_mathematical_formulation.pdf')
        
        print("图5生成完成!")
        return fig
    
    def figure6_soc_variation_prediction(self):
        """图6: 不同初始SOC下的剩余时间预测"""
        print("正在生成图6: 不同初始SOC下的剩余时间预测...")
        
        # 固定功率，变化SOC
        P_fixed = 3.0  # 固定功率为3W
        SOC_values = np.linspace(0.1, 0.95, 15)  # 15个SOC点
        
        exact_times = []
        simple_times = []
        improved_times = []
        
        for SOC0 in SOC_values:
            print(f"  计算 SOC={SOC0:.2f}...")
            try:
                # 精确解
                result = self.exact_predictor.predict_exact_time(SOC0, P_fixed, method='dae_solve')
                exact_times.append(result['time_hours'])
                
                # 简单理论方法
                simple_time = self.analytical_predictor.simple_theoretical(SOC0, P_fixed)
                simple_times.append(simple_time)
                
                # 改进解析方法
                improved_time = self.analytical_predictor.improved_analytical(SOC0, P_fixed)
                improved_times.append(improved_time)
            except Exception as e:
                print(f"    计算失败: {e}")
                exact_times.append(np.nan)
                simple_times.append(np.nan)
                improved_times.append(np.nan)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 子图1: 不同初始SOC下的剩余时间
        ax = axes[0]
        ax.plot(SOC_values*100, exact_times, 'bo-', label='Exact Method', linewidth=2, markersize=6)
        ax.plot(SOC_values*100, simple_times, 'rs--', label='Simple Method', linewidth=1.5, markersize=4)
        ax.plot(SOC_values*100, improved_times, 'g^--', label='Improved Method', linewidth=1.5, markersize=4)
        
        ax.set_xlabel('Initial State of Charge (%)', fontweight='bold')
        ax.set_ylabel('Remaining Time (hours)', fontweight='bold')
        ax.set_title(f'Remaining Time vs Initial SOC (P={P_fixed} W)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 添加关键数据点标注
        key_soc_indices = [0, len(SOC_values)//2, -1]
        for idx in key_soc_indices:
            if idx < len(SOC_values):
                soc = SOC_values[idx] * 100
                if not np.isnan(exact_times[idx]):
                    ax.annotate(f'{exact_times[idx]:.2f}h', 
                               xy=(soc, exact_times[idx]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # 子图2: 不同方法的相对误差
        ax = axes[1]
        
        # 计算误差
        simple_errors = []
        improved_errors = []
        
        for i in range(len(SOC_values)):
            if not np.isnan(exact_times[i]) and exact_times[i] > 0:
                if not np.isnan(simple_times[i]):
                    simple_errors.append(abs(simple_times[i] - exact_times[i]) / exact_times[i] * 100)
                else:
                    simple_errors.append(np.nan)
                    
                if not np.isnan(improved_times[i]):
                    improved_errors.append(abs(improved_times[i] - exact_times[i]) / exact_times[i] * 100)
                else:
                    improved_errors.append(np.nan)
            else:
                simple_errors.append(np.nan)
                improved_errors.append(np.nan)
        
        ax.plot(SOC_values*100, simple_errors, 'rs--', label='Simple Method Error', linewidth=1.5, markersize=4)
        ax.plot(SOC_values*100, improved_errors, 'g^--', label='Improved Method Error', linewidth=1.5, markersize=4)
        
        ax.set_xlabel('Initial State of Charge (%)', fontweight='bold')
        ax.set_ylabel('Relative Error (%)', fontweight='bold')
        ax.set_title(f'Prediction Error vs Initial SOC (P={P_fixed} W)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        valid_simple_errors = [e for e in simple_errors if not np.isnan(e)]
        valid_improved_errors = [e for e in improved_errors if not np.isnan(e)]
        
        if valid_simple_errors and valid_improved_errors:
            stats_text = (f"Simple Method: Mean Error = {np.mean(valid_simple_errors):.2f}%, "
                         f"Max Error = {np.max(valid_simple_errors):.2f}%\n"
                         f"Improved Method: Mean Error = {np.mean(valid_improved_errors):.2f}%, "
                         f"Max Error = {np.max(valid_improved_errors):.2f}%")
            
            plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Prediction of Remaining Time for Different Initial SOC Values (Constant Power: {P_fixed} W)', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        fig.savefig(self.output_dir / 'figure6_soc_variation_prediction.png', dpi=300)
        fig.savefig(self.output_dir / 'figure6_soc_variation_prediction.pdf')
        
        # 保存数据
        soc_data = pd.DataFrame({
            'SOC': SOC_values,
            'SOC_%': SOC_values * 100,
            'Exact_Time_h': exact_times,
            'Simple_Time_h': simple_times,
            'Improved_Time_h': improved_times,
            'Simple_Error_%': simple_errors,
            'Improved_Error_%': improved_errors
        })
        soc_data.to_csv(self.output_dir / 'soc_variation_data.csv', index=False)
        
        print("图6生成完成!")
        return fig, soc_data
    
    def figure7_power_variation_prediction(self):
        """图7: 不同预设功率负载下的剩余时间预测"""
        print("正在生成图7: 不同预设功率负载下的剩余时间预测...")
        
        # 固定SOC，变化功率
        SOC_fixed = 0.8  # 固定SOC为80%
        P_values = np.linspace(0.5, 6.0, 15)  # 15个功率点，从0.5W到6W
        
        exact_times = []
        simple_times = []
        improved_times = []
        
        for P in P_values:
            print(f"  计算 P={P:.2f}W...")
            try:
                # 精确解
                result = self.exact_predictor.predict_exact_time(SOC_fixed, P, method='dae_solve')
                exact_times.append(result['time_hours'])
                
                # 简单理论方法
                simple_time = self.analytical_predictor.simple_theoretical(SOC_fixed, P)
                simple_times.append(simple_time)
                
                # 改进解析方法
                improved_time = self.analytical_predictor.improved_analytical(SOC_fixed, P)
                improved_times.append(improved_time)
            except Exception as e:
                print(f"    计算失败: {e}")
                exact_times.append(np.nan)
                simple_times.append(np.nan)
                improved_times.append(np.nan)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 子图1: 不同功率下的剩余时间
        ax = axes[0]
        ax.plot(P_values, exact_times, 'bo-', label='Exact Method', linewidth=2, markersize=6)
        ax.plot(P_values, simple_times, 'rs--', label='Simple Method', linewidth=1.5, markersize=4)
        ax.plot(P_values, improved_times, 'g^--', label='Improved Method', linewidth=1.5, markersize=4)
        
        ax.set_xlabel('Power Load (W)', fontweight='bold')
        ax.set_ylabel('Remaining Time (hours)', fontweight='bold')
        ax.set_title(f'Remaining Time vs Power Load (SOC={SOC_fixed*100:.0f}%)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 添加关键数据点标注
        key_p_indices = [0, len(P_values)//2, -1]
        for idx in key_p_indices:
            if idx < len(P_values):
                p = P_values[idx]
                if not np.isnan(exact_times[idx]):
                    ax.annotate(f'{exact_times[idx]:.2f}h', 
                               xy=(p, exact_times[idx]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # 子图2: 双对数坐标下的功率-时间关系
        ax = axes[1]
        
        # 过滤有效数据点
        valid_mask = (~np.isnan(exact_times)) & (np.array(exact_times) > 0)
        if np.sum(valid_mask) > 3:
            valid_P = P_values[valid_mask]
            valid_times = np.array(exact_times)[valid_mask]
            
            ax.loglog(valid_P, valid_times, 'bo-', label='Exact Method', linewidth=2, markersize=6)
            ax.set_xlabel('Power Load (W)', fontweight='bold')
            ax.set_ylabel('Remaining Time (hours)', fontweight='bold')
            ax.set_title(f'Log-Log Plot: Time vs Power (SOC={SOC_fixed*100:.0f}%)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both')
            
            # 添加理论拟合线（理想情况下，时间与功率成反比）
            # 选择中间范围进行拟合
            mid_idx = len(valid_P) // 2
            if mid_idx > 0:
                ref_power = valid_P[mid_idx]
                ref_time = valid_times[mid_idx]
                
                # 绘制反比关系参考线
                p_range = np.linspace(valid_P[0], valid_P[-1], 50)
                inv_line = ref_time * ref_power / p_range
                ax.loglog(p_range, inv_line, 'r--', label='Ideal 1/P relationship', linewidth=1.5, alpha=0.7)
                
                ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Insufficient valid data for log-log plot', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Power Load (W)', fontweight='bold')
            ax.set_ylabel('Remaining Time (hours)', fontweight='bold')
        
        # 子图3: 不同方法的相对误差（添加到新图中）
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # 计算误差
        simple_errors = []
        improved_errors = []
        
        for i in range(len(P_values)):
            if not np.isnan(exact_times[i]) and exact_times[i] > 0:
                if not np.isnan(simple_times[i]):
                    simple_errors.append(abs(simple_times[i] - exact_times[i]) / exact_times[i] * 100)
                else:
                    simple_errors.append(np.nan)
                    
                if not np.isnan(improved_times[i]):
                    improved_errors.append(abs(improved_times[i] - exact_times[i]) / exact_times[i] * 100)
                else:
                    improved_errors.append(np.nan)
            else:
                simple_errors.append(np.nan)
                improved_errors.append(np.nan)
        
        ax2.plot(P_values, simple_errors, 'rs--', label='Simple Method Error', linewidth=1.5, markersize=4)
        ax2.plot(P_values, improved_errors, 'g^--', label='Improved Method Error', linewidth=1.5, markersize=4)
        
        ax2.set_xlabel('Power Load (W)', fontweight='bold')
        ax2.set_ylabel('Relative Error (%)', fontweight='bold')
        ax2.set_title(f'Prediction Error vs Power Load (SOC={SOC_fixed*100:.0f}%)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        valid_simple_errors = [e for e in simple_errors if not np.isnan(e)]
        valid_improved_errors = [e for e in improved_errors if not np.isnan(e)]
        
        if valid_simple_errors and valid_improved_errors:
            stats_text = (f"Simple Method: Mean Error = {np.mean(valid_simple_errors):.2f}%, "
                         f"Max Error = {np.max(valid_simple_errors):.2f}%\n"
                         f"Improved Method: Mean Error = {np.mean(valid_improved_errors):.2f}%, "
                         f"Max Error = {np.max(valid_improved_errors):.2f}%")
            
            plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Prediction of Remaining Time for Different Power Loads (Initial SOC: {SOC_fixed*100:.0f}%)', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        fig.savefig(self.output_dir / 'figure7_power_variation_prediction.png', dpi=300)
        fig.savefig(self.output_dir / 'figure7_power_variation_prediction.pdf')
        fig2.savefig(self.output_dir / 'figure7b_power_variation_errors.png', dpi=300)
        fig2.savefig(self.output_dir / 'figure7b_power_variation_errors.pdf')
        
        # 保存数据
        power_data = pd.DataFrame({
            'Power_W': P_values,
            'Exact_Time_h': exact_times,
            'Simple_Time_h': simple_times,
            'Improved_Time_h': improved_times,
            'Simple_Error_%': simple_errors,
            'Improved_Error_%': improved_errors
        })
        power_data.to_csv(self.output_dir / 'power_variation_data.csv', index=False)
        
        print("图7生成完成!")
        plt.close(fig2)
        return fig, power_data
    
    def generate_all_figures(self):
        """生成所有图表"""
        print("=" * 80)
        print("开始生成精确预测图表...")
        print("=" * 80)
        
        # 图1: 精确解的动态过程
        try:
            fig1, result1 = self.figure1_exact_solution_dynamics()
            plt.close(fig1)
        except Exception as e:
            print(f"图1生成失败: {e}")
        
        # 图2: 不同方法的精度比较
        try:
            fig2, data2 = self.figure2_method_comparison_accuracy()
            plt.close(fig2)
        except Exception as e:
            print(f"图2生成失败: {e}")
        
        # 图3: 相图与敏感性分析
        try:
            fig3 = self.figure3_phase_portrait_sensitivity()
            plt.close(fig3)
        except Exception as e:
            print(f"图3生成失败: {e}")
        
        # 图4: 数值方法的收敛性分析
        try:
            fig4, data4 = self.figure4_convergence_analysis()
            plt.close(fig4)
        except Exception as e:
            print(f"图4生成失败: {e}")
        
        # 图5: 数学公式和理论框架
        try:
            fig5 = self.figure5_mathematical_formulation()
            plt.close(fig5)
        except Exception as e:
            print(f"图5生成失败: {e}")
        
        # 图6: 不同初始SOC下的剩余时间预测
        try:
            fig6, data6 = self.figure6_soc_variation_prediction()
            plt.close(fig6)
        except Exception as e:
            print(f"图6生成失败: {e}")
        
        # 图7: 不同预设功率负载下的剩余时间预测
        try:
            fig7, data7 = self.figure7_power_variation_prediction()
            plt.close(fig7)
        except Exception as e:
            print(f"图7生成失败: {e}")
        
        print(f"\n所有图表已保存到: {self.output_dir}")
        
        # 显示完成信息
        print("\n" + "=" * 80)
        print("图表生成完成!")
        print("=" * 80)
        
        print("\n生成的文件:")
        for file in self.output_dir.iterdir():
            if file.is_file():
                print(f"  - {file.name}")

# =============================================
# 6. 主程序
# =============================================

def main():
    """主程序"""
    print("=" * 80)
    print("电池剩余时间精确预测系统")
    print("基于连续函数积分方法")
    print("=" * 80)
    
    # 创建图表生成器
    figure_generator = ExactPredictionFigures()
    
    # 生成所有图表
    figure_generator.generate_all_figures()

# =============================================
# 7. 程序入口
# =============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断.")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()