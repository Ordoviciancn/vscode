import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os

# Create output directory
output_dir = "battery_figures_with_soc_dependent_dcr"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# =============================================================================
# 1. SOC-Dependent DCR Interpolation Function
# =============================================================================

# SOC-DCR data at 20°C (20 points, equal spacing)
soc_20c = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00])
dcr_20c = np.array([0.019100, 0.018450, 0.017850, 0.017250, 0.016600, 0.016000,
                    0.015500, 0.015100, 0.014700, 0.014350, 0.014050, 0.013750,
                    0.013450, 0.013200, 0.012950, 0.012700, 0.012500, 0.012350,
                    0.012180, 0.012000])

# Build linear interpolation function
dcr_interp_20c = interp1d(
    soc_20c, dcr_20c,
    kind='linear',
    fill_value=(dcr_20c[0], dcr_20c[-1]),
    bounds_error=False
)

# =============================================================================
# 2. Battery Model Parameters
# =============================================================================

# Assume battery: 3000mAh, nominal 3.7V, energy ~11.1Wh
Q_capacity_Ah = 3.0 
Q_capacity_C = Q_capacity_Ah * 3600  # Coulombs

# OCV-SOC curve (based on empirical formula)
def get_ocv(soc):
    # Limit soc to 0-1 to prevent polynomial divergence outside physical range
    soc = np.clip(soc, 0, 1)
    return (66.939 * soc**7 - 248.739 * soc**6 + 371.911 * soc**5 
            - 291.517 * soc**4 + 132.659 * soc**3 - 36.387 * soc**2 
            + 6.069 * soc + 3.259)

# Physical parameters (RC circuit parameters remain unchanged)
# R0 is now calculated dynamically via interpolation function
R1 = 0.003; C1 = 4850
R2 = 0.002; C2 = 88200
V_cutoff = 3.0  # Cutoff voltage

# =============================================================================
# 3. Core Dynamics (with SOC-dependent R0 integration)
# =============================================================================
def battery_dynamics(t, y, P_load):
    soc, vp1, vp2 = y
    
    if soc <= 0: 
        return [0, 0, 0]  # Protection
    
    # 1. Get dynamic R0 based on current SOC
    R0 = dcr_interp_20c(soc)
    
    # 2. Calculate OCV
    ocv = get_ocv(soc)
    
    # 3. Solve algebraic loop: V_term^2 - (OCV-Vp1-Vp2)*V_term + P*R0 = 0
    U_effective = ocv - vp1 - vp2
    delta = U_effective**2 - 4 * P_load * R0
    
    if delta < 0:
        # Power too high, voltage collapse
        v_term = V_cutoff
        i_batt = P_load / v_term
    else:
        v_term = (U_effective + np.sqrt(delta)) / 2
        i_batt = P_load / v_term
    
    # 4. Calculate state derivatives
    d_soc = -i_batt / Q_capacity_C
    d_vp1 = (i_batt/C1) - (vp1/(R1*C1))
    d_vp2 = (i_batt/C2) - (vp2/(R2*C2))
    
    return [d_soc, d_vp1, d_vp2]

# =============================================================================
# 4. Enhanced Solver (with resistance tracking)
# =============================================================================
def simulate_discharge(P_load, soc_init=1.0):
    t_span = [0, 50 * 3600]  # Maximum 50 hours
    y0 = [soc_init, 0, 0]
    
    # Termination event: voltage below cutoff
    def voltage_cutoff(t, y):
        soc, vp1, vp2 = y
        R0 = dcr_interp_20c(soc)  # Dynamic R0
        U_eff = get_ocv(soc) - vp1 - vp2
        delta = U_eff**2 - 4 * P_load * R0
        if delta < 0: 
            return -1.0
        v_term = (U_eff + np.sqrt(delta)) / 2
        return v_term - V_cutoff
    
    voltage_cutoff.terminal = True
    voltage_cutoff.direction = -1
    
    # Termination event: SOC below 0
    def soc_cutoff(t, y):
        return y[0]
    soc_cutoff.terminal = True
    
    # Solve ODE
    sol = solve_ivp(
        lambda t, y: battery_dynamics(t, y, P_load),
        t_span, 
        y0, 
        events=[voltage_cutoff, soc_cutoff],
        method='RK45', 
        rtol=1e-4, 
        atol=1e-6, 
        max_step=60
    )
    
    # Reconstruct detailed outputs
    time_h = sol.t / 3600
    soc_traj = sol.y[0]
    vp1_traj = sol.y[1]
    vp2_traj = sol.y[2]
    
    v_traj = []
    i_traj = []
    r0_traj = []
    
    for i, s in enumerate(soc_traj):
        # Dynamic R0 calculation
        R0 = dcr_interp_20c(s)
        r0_traj.append(R0)
        
        U_eff = get_ocv(s) - vp1_traj[i] - vp2_traj[i]
        delta = U_eff**2 - 4 * P_load * R0
        
        if delta < 0: 
            v_term = V_cutoff
            i_batt = P_load / v_term
        else:
            v_term = (U_eff + np.sqrt(delta)) / 2
            i_batt = P_load / v_term
            
        v_traj.append(v_term)
        i_traj.append(i_batt)
    
    tte = time_h[-1] if sol.status == 1 else time_h[-1]
    
    return time_h, soc_traj, np.array(v_traj), np.array(i_traj), np.array(r0_traj), tte

# =============================================================================
# 5. Plotting Settings (no font specification)
# =============================================================================
# Using default matplotlib fonts - no font specification
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 10

# =============================================================================
# Figure 1: DCR-SOC Interpolation Function Validation
# =============================================================================
fig0 = plt.figure(figsize=(8, 5))
ax0 = fig0.add_subplot(1, 1, 1)

# Plot original data points
ax0.scatter(soc_20c, dcr_20c * 1000, color='red', s=60, zorder=5, label='Experimental Data Points', edgecolors='black')

# Plot interpolation curve
soc_test = np.linspace(0, 1, 200)
dcr_test = dcr_interp_20c(soc_test) * 1000  # Convert to mΩ
ax0.plot(soc_test, dcr_test, 'b-', linewidth=2, label='Linear Interpolation Curve')

ax0.set_xlabel('SOC')
ax0.set_ylabel('DCR (mΩ)')
ax0.set_title('SOC-DCR Relationship (20°C)')
ax0.grid(True, linestyle=':', alpha=0.6)
ax0.legend(loc='upper right', frameon=True)
ax0.set_xlim(0, 1)
ax0.set_ylim(11, 20)

# Add key point annotations
for soc, dcr in zip(soc_20c[::2], dcr_20c[::2]):
    ax0.annotate(f'{dcr*1000:.1f}mΩ', 
                xy=(soc, dcr*1000), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8)

plt.tight_layout()
plt.savefig(f'{output_dir}/Figure_0_DCR_SOC_Relation.png', dpi=300)
plt.close(fig0)

# =============================================================================
# Figure 2: Discharge Curves under Different Power Loads
# =============================================================================
fig1 = plt.figure(figsize=(9, 6))
ax1a = fig1.add_subplot(2, 1, 1)
ax1b = fig1.add_subplot(2, 1, 2)

powers = [0.5, 1.5, 3.0, 10.0]
colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

for p, c in zip(powers, colors):
    t, s, v, i, r0, tte = simulate_discharge(p)
    
    # Upper subplot: Voltage curves
    ax1a.plot(t, v, label=f'P={p}W (TTE={tte:.1f}h)', color=c, linewidth=2)
    
    # Lower subplot: Internal resistance variation
    ax1b.plot(t, r0*1000, color=c, linewidth=2, alpha=0.7)

ax1a.axhline(V_cutoff, color='k', linestyle='--', linewidth=1, label='Cutoff Voltage')
ax1a.set_xlabel('Time (Hours)')
ax1a.set_ylabel('Terminal Voltage (V)')
ax1a.set_title('Voltage Discharge Profiles under Various Power Loads (with SOC-Dependent DCR)')
ax1a.legend(frameon=True, fontsize=9, loc='upper right')
ax1a.grid(True, linestyle=':', alpha=0.6)
ax1a.set_xlim(left=0)

ax1b.set_xlabel('Time (Hours)')
ax1b.set_ylabel('Dynamic Resistance R0 (mΩ)')
ax1b.set_title('Internal Resistance Variation under Different Power Loads')
ax1b.grid(True, linestyle=':', alpha=0.6)
ax1b.set_xlim(left=0)

plt.tight_layout()
plt.savefig(f'{output_dir}/Figure_a_Voltage_Profiles_with_DCR.png', dpi=300)
plt.close(fig1)

# =============================================================================
# Figure 3: Detailed Dynamics Analysis under 3W Load (Four Subplots)
# =============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
ax2a, ax2b, ax2c, ax2d = axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]

# Simulate 3W discharge
t, s, v, i, r0, tte = simulate_discharge(3.0)

# Subplot a: Dual-axis SOC and Voltage
ax2a.plot(t, s*100, color='#8e44ad', linewidth=2, label='SOC (%)')
ax2a.set_ylabel('State of Charge (%)', color='#8e44ad')
ax2a.tick_params(axis='y', labelcolor='#8e44ad')
ax2a.set_xlabel('Time (Hours)')

ax2a_r = ax2a.twinx()
ax2a_r.plot(t, v, color='#2c3e50', linewidth=2, linestyle='-', label='Voltage (V)')
ax2a_r.set_ylabel('Terminal Voltage (V)', color='#2c3e50')
ax2a_r.tick_params(axis='y', labelcolor='#2c3e50')

lines1, labels1 = ax2a.get_legend_handles_labels()
lines2, labels2 = ax2a_r.get_legend_handles_labels()
ax2a.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

# Subplot b: Current and Power
ax2b.plot(t, i*1000, color='#e74c3c', linewidth=2, label='Current (mA)')
ax2b.set_xlabel('Time (Hours)')
ax2b.set_ylabel('Current (mA)', color='#e74c3c')
ax2b.tick_params(axis='y', labelcolor='#e74c3c')
ax2b_r = ax2b.twinx()
ax2b_r.plot(t, 3.0*np.ones_like(t), color='#3498db', linewidth=2, linestyle='--', label='Power (W)')
ax2b_r.set_ylabel('Power (W)', color='#3498db')
ax2b_r.tick_params(axis='y', labelcolor='#3498db')
ax2b.legend(loc='upper right', frameon=True)

# Subplot c: Dynamic Resistance Variation
ax2c.plot(t, r0*1000, color='#27ae60', linewidth=2)
ax2c.fill_between(t, r0*1000*0.98, r0*1000*1.02, color='#27ae60', alpha=0.3)
ax2c.set_xlabel('Time (Hours)')
ax2c.set_ylabel('Dynamic Resistance R0 (mΩ)')
ax2c.set_title(f'Dynamic Resistance Variation (Initial: {r0[0]*1000:.2f}mΩ, Final: {r0[-1]*1000:.2f}mΩ)')
ax2c.grid(True, linestyle=':', alpha=0.6)

# Subplot d: Resistance vs SOC Relationship
ax2d.scatter(s*100, r0*1000, c=t, cmap='viridis', s=30, edgecolors='k', linewidth=0.5)
ax2d.set_xlabel('SOC (%)')
ax2d.set_ylabel('Dynamic Resistance R0 (mΩ)')
ax2d.set_title('Relationship between Internal Resistance and SOC')
ax2d.grid(True, linestyle=':', alpha=0.6)

# Add colorbar
cbar = plt.colorbar(ax2d.collections[0], ax=ax2d, orientation='vertical', pad=0.02)
cbar.set_label('Time (Hours)', rotation=270, labelpad=15)

for ax in [ax2a, ax2b, ax2c, ax2d]:
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(f'{output_dir}/Figure_b_Detailed_Dynamics_3W.png', dpi=300)
plt.close(fig2)

# =============================================================================
# Figure 4: TTE Heatmap (Comparison of Fixed R0 vs Dynamic R0)
# =============================================================================
fig3 = plt.figure(figsize=(10, 8))

# Subplot a: Fixed R0 = 0.012Ω
ax3a = fig3.add_subplot(2, 2, 1)

# Temporary function: simulation with fixed R0
def simulate_fixed_r0(P_load, soc_init=1.0):
    # Temporary copy of battery dynamics function, but with fixed R0
    def battery_dynamics_fixed(t, y, P_load):
        soc, vp1, vp2 = y
        if soc <= 0: return [0, 0, 0]
        
        # Fixed R0 = 0.012Ω
        R0 = 0.012
        ocv = get_ocv(soc)
        U_effective = ocv - vp1 - vp2
        delta = U_effective**2 - 4 * P_load * R0
        
        if delta < 0:
            v_term = V_cutoff
            i_batt = P_load / v_term
        else:
            v_term = (U_effective + np.sqrt(delta)) / 2
            i_batt = P_load / v_term
        
        d_soc = -i_batt / Q_capacity_C
        d_vp1 = (i_batt/C1) - (vp1/(R1*C1))
        d_vp2 = (i_batt/C2) - (vp2/(R2*C2))
        
        return [d_soc, d_vp1, d_vp2]
    
    # Solve (simplified version)
    sol = solve_ivp(
        lambda t,y: battery_dynamics_fixed(t,y,P_load),
        [0, 50*3600], 
        [soc_init, 0, 0],
        method='RK45', 
        max_step=60
    )
    
    return sol.t[-1] / 3600

# Calculate TTE heatmap for fixed R0
soc_vals = np.linspace(0.1, 1.0, 10)
p_vals = np.linspace(0.5, 5.0, 10)
X, Y = np.meshgrid(p_vals, soc_vals)
Z_fixed = np.zeros_like(X)

for i in range(len(soc_vals)):
    for j in range(len(p_vals)):
        Z_fixed[i, j] = simulate_fixed_r0(p_vals[j], soc_init=soc_vals[i])

cp1 = ax3a.contourf(X, Y, Z_fixed, levels=20, cmap='YlOrRd')
plt.colorbar(cp1, ax=ax3a).set_label('TTE (Hours)', rotation=270, labelpad=15)
ax3a.set_xlabel('Power Load (W)')
ax3a.set_ylabel('Initial SOC')
ax3a.set_title('TTE Prediction with Fixed R0=12mΩ')

# Subplot b: TTE heatmap with dynamic R0
ax3b = fig3.add_subplot(2, 2, 2)

Z_dynamic = np.zeros_like(X)
for i in range(len(soc_vals)):
    for j in range(len(p_vals)):
        _, _, _, _, _, tte = simulate_discharge(p_vals[j], soc_init=soc_vals[i])
        Z_dynamic[i, j] = tte

cp2 = ax3b.contourf(X, Y, Z_dynamic, levels=20, cmap='YlOrRd')
plt.colorbar(cp2, ax=ax3b).set_label('TTE (Hours)', rotation=270, labelpad=15)
ax3b.set_xlabel('Power Load (W)')
ax3b.set_ylabel('Initial SOC')
ax3b.set_title('TTE Prediction with Dynamic R0')

# Subplot c: TTE difference (dynamic - fixed)
ax3c = fig3.add_subplot(2, 2, 3)
Z_diff = Z_dynamic - Z_fixed
cp3 = ax3c.contourf(X, Y, Z_diff, levels=20, cmap='RdBu_r')
plt.colorbar(cp3, ax=ax3c).set_label('TTE Difference (Hours)', rotation=270, labelpad=15)
ax3c.set_xlabel('Power Load (W)')
ax3c.set_ylabel('Initial SOC')
ax3c.set_title('TTE Prediction Difference (Dynamic R0 - Fixed R0)')

# Subplot d: Percentage difference
ax3d = fig3.add_subplot(2, 2, 4)
Z_pct_diff = 100 * (Z_dynamic - Z_fixed) / Z_fixed
cp4 = ax3d.contourf(X, Y, Z_pct_diff, levels=20, cmap='RdBu_r')
plt.colorbar(cp4, ax=ax3d).set_label('Difference Percentage (%)', rotation=270, labelpad=15)
ax3d.set_xlabel('Power Load (W)')
ax3d.set_ylabel('Initial SOC')
ax3d.set_title('TTE Percentage Difference')

plt.tight_layout()
plt.savefig(f'{output_dir}/Figure_c_TTE_Comparison.png', dpi=300)
plt.close(fig3)

# =============================================================================
# Figure 5: Monte Carlo Uncertainty Analysis (with Power Fluctuations)
# =============================================================================
fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(1, 1, 1)

tte_dist_dynamic = []
r0_end_dist = []

for _ in range(300):
    # Simulate power fluctuations: mean 3W, std 0.5W
    p_rand = np.random.normal(3.0, 0.5)
    if p_rand < 0.1: 
        p_rand = 0.1
    
    # Simulate discharge
    _, _, _, _, r0_traj, tte = simulate_discharge(p_rand, soc_init=0.8)
    tte_dist_dynamic.append(tte)
    r0_end_dist.append(r0_traj[-1]*1000)  # Record final R0 value

# Plot distribution
sns.histplot(tte_dist_dynamic, kde=True, color='#34495e', element="step", ax=ax4, bins=20)
mu = np.mean(tte_dist_dynamic)
ci_low = np.percentile(tte_dist_dynamic, 2.5)
ci_high = np.percentile(tte_dist_dynamic, 97.5)

ax4.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'Mean: {mu:.2f}h')
ax4.axvspan(ci_low, ci_high, color='orange', alpha=0.2, label=f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]h')

ax4.set_xlabel('Discharge Time (Hours)')
ax4.set_ylabel('Frequency')
ax4.set_title('Discharge Time Distribution under Power Fluctuations (Initial SOC=0.8, Dynamic R0)')
ax4.legend(loc='upper right', fontsize=9)

# Add resistance information
ax4.text(0.05, 0.95, f'Final Resistance Range: {np.min(r0_end_dist):.1f}-{np.max(r0_end_dist):.1f}mΩ',
         transform=ax4.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/Figure_d_Uncertainty_Analysis.png', dpi=300)
plt.close(fig4)

# =============================================================================
# Results Output
# =============================================================================
print("="*70)
print("Battery Model Simulation Complete (with SOC-Dependent Internal Resistance)")
print("="*70)
print(f"All figures saved to '{output_dir}' directory:")
print(f"1. {output_dir}/Figure_0_DCR_SOC_Relation.png - SOC-DCR Relationship Validation")
print(f"2. {output_dir}/Figure_a_Voltage_Profiles_with_DCR.png - Voltage and Resistance Curves")
print(f"3. {output_dir}/Figure_b_Detailed_Dynamics_3W.png - Detailed Dynamics under 3W Load")
print(f"4. {output_dir}/Figure_c_TTE_Comparison.png - TTE Heatmap Comparison")
print(f"5. {output_dir}/Figure_d_Uncertainty_Analysis.png - Uncertainty Analysis")
print("\nKey Findings:")
print(f"- DCR Range: {dcr_20c[0]*1000:.1f}mΩ (SOC=0) → {dcr_20c[-1]*1000:.1f}mΩ (SOC=1)")
print(f"- Resistance Change Rate: {(dcr_20c[0]/dcr_20c[-1]-1)*100:.1f}%")
print(f"- Resistance Variation during 3W Discharge: {dcr_interp_20c(1.0)*1000:.1f}mΩ → {dcr_interp_20c(0.0)*1000:.1f}mΩ")
print("="*70)