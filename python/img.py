import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ==================== BATTERY MODEL PARAMETERS ====================
class BatteryParameters:
    def __init__(self):
        # Battery electrochemical parameters
        self.Cn = 3.0  # Nominal capacity (Ah)
        self.SOH = 1.0  # State of Health
        self.eta = 1.0  # Coulombic efficiency (discharge)
        self.eta_conv = 0.95  # Power conversion efficiency
        
        # Second-order RC model parameters
        self.R0 = 0.05   # Ohmic resistance (Ω)
        self.R1 = 0.01   # Electrochemical polarization resistance (Ω)
        self.C1 = 2000.0 # Electrochemical polarization capacitance (F)
        self.R2 = 0.02   # Concentration polarization resistance (Ω)
        self.C2 = 5000.0 # Concentration polarization capacitance (F)
        
        # OCV-SOC polynomial coefficients (7th order fit)
        self.ocv_coeffs = np.array([
            3.259, 6.069, -36.387, 132.659, -291.517, 
            371.911, -248.739, 66.939
        ])
    
    def V_oc(self, SOC):
        """Calculate open-circuit voltage"""
        return np.polyval(self.ocv_coeffs, SOC)

# ==================== POWER CONSUMPTION MODEL ====================
class PowerModel:
    def __init__(self):
        # Base power consumption (mW) - minimum operating power
        self.base_power = 150.0  # Reduced from 200mW
        
        # Hardware component power coefficients (mW per unit)
        # Based on literature data and actual measurements
        self.power_coeffs = {
            # CPU power (dynamic + static)
            'cpu': {
                'idle': 150,           # Reduced idle power
                'light': 12.0,         # Light load (mW/% util)
                'medium': 20.0,        # Medium load (mW/% util)
                'heavy': 35.0          # Heavy load (mW/% util)
            },
            
            # GPU power (mobile GPU)
            'gpu': {
                'idle': 40,            # Idle state (mW)
                'light': 8.0,          # Light load (mW/% util)
                'heavy': 25.0          # Heavy load (mW/% util)
            },
            
            # Screen power (AMOLED display)
            'screen': {
                'per_brightness': 3.5,  # Per 1% brightness (mW) - reduced
                'always_on': 80         # Reduced always-on display
            },
            
            # Cellular network
            'cellular': {
                '4g_idle': 250,         # Reduced 4G standby
                '4g_active': 700,       # Reduced 4G active
                '5g_idle': 400,         # Reduced 5G standby
                '5g_active': 950,       # Reduced 5G active
                'download': 22.0,       # Download traffic (mW/MB/s)
                'upload': 25.0          # Upload traffic (mW/MB/s)
            },
            
            # WiFi
            'wifi': {
                'idle': 120,           # Reduced WiFi standby
                'active': 250,         # Reduced WiFi active
                'download': 0.07,      # Reduced download traffic
                'upload': 0.10         # Reduced upload traffic
            },
            
            # GPS
            'gps': {
                'cold_start': 400,     # Reduced cold start
                'tracking': 120        # Reduced continuous tracking
            },
            
            # Camera module
            'camera': {
                'preview': 700,        # Reduced preview mode
                'photo': 1300,         # Reduced photo capture
                'video_1080p': 1000,   # Reduced 1080p video recording
                'video_4k': 1800,      # Reduced 4K video recording
                'flash': 1500          # Reduced flash
            },
            
            # Audio
            'audio': {
                'speaker': 400,        # Reduced speaker
                'headphone': 80,       # Reduced headphone
                'volume_factor': 8.5   # Reduced volume coefficient
            },
            
            # Memory and storage
            'memory': {'active': 40},      # Reduced memory active
            'storage': {'active': 80},     # Reduced storage active
            
            # Other components
            'bluetooth': {'active': 150},  # Reduced Bluetooth
            'sensors': {'active': 40},     # Reduced sensors
            'vibration': {'active': 250}   # Reduced vibration motor
        }
    
    def calculate_power(self, usage_mode):
        """
        Calculate total power consumption for a specific usage scenario (mW)
        usage_mode: dictionary containing detailed hardware component states
        """
        total_power = self.base_power
        
        # CPU power consumption
        cpu_mode = usage_mode.get('cpu_mode', 'light')
        cpu_util = usage_mode.get('cpu_util', 10)
        
        if cpu_mode == 'idle':
            total_power += self.power_coeffs['cpu']['idle']
        elif cpu_mode == 'light':
            total_power += self.power_coeffs['cpu']['light'] * cpu_util
        elif cpu_mode == 'medium':
            total_power += self.power_coeffs['cpu']['medium'] * cpu_util
        elif cpu_mode == 'heavy':
            total_power += self.power_coeffs['cpu']['heavy'] * cpu_util
        
        # GPU power consumption
        if usage_mode.get('gpu_active', False):
            gpu_mode = usage_mode.get('gpu_mode', 'light')
            gpu_util = usage_mode.get('gpu_util', 10)
            
            if gpu_mode == 'idle':
                total_power += self.power_coeffs['gpu']['idle']
            elif gpu_mode == 'light':
                total_power += self.power_coeffs['gpu']['light'] * gpu_util
            elif gpu_mode == 'heavy':
                total_power += self.power_coeffs['gpu']['heavy'] * gpu_util
        
        # Screen power consumption
        if usage_mode.get('screen_on', False):
            brightness = usage_mode.get('screen_brightness', 50)
            total_power += self.power_coeffs['screen']['per_brightness'] * brightness
            
            if usage_mode.get('always_on_display', False):
                total_power += self.power_coeffs['screen']['always_on']
        
        # Cellular network
        cellular_type = usage_mode.get('cellular_type', 'none')
        if cellular_type == '4g':
            if usage_mode.get('cellular_active', False):
                total_power += self.power_coeffs['cellular']['4g_active']
                
                # Traffic power consumption
                download_rate = usage_mode.get('download_rate', 0)
                upload_rate = usage_mode.get('upload_rate', 0)
                total_power += self.power_coeffs['cellular']['download'] * download_rate
                total_power += self.power_coeffs['cellular']['upload'] * upload_rate
            else:
                total_power += self.power_coeffs['cellular']['4g_idle']
        elif cellular_type == '5g':
            if usage_mode.get('cellular_active', False):
                total_power += self.power_coeffs['cellular']['5g_active']
                
                # Traffic power consumption
                download_rate = usage_mode.get('download_rate', 0)
                upload_rate = usage_mode.get('upload_rate', 0)
                total_power += self.power_coeffs['cellular']['download'] * download_rate
                total_power += self.power_coeffs['cellular']['upload'] * upload_rate
            else:
                total_power += self.power_coeffs['cellular']['5g_idle']
        
        # WiFi
        if usage_mode.get('wifi_on', False):
            if usage_mode.get('wifi_active', False):
                total_power += self.power_coeffs['wifi']['active']
                
                # Traffic power consumption
                wifi_download = usage_mode.get('wifi_download', 0)
                wifi_upload = usage_mode.get('wifi_upload', 0)
                total_power += self.power_coeffs['wifi']['download'] * wifi_download
                total_power += self.power_coeffs['wifi']['upload'] * wifi_upload
            else:
                total_power += self.power_coeffs['wifi']['idle']
        
        # GPS
        if usage_mode.get('gps_on', False):
            gps_mode = usage_mode.get('gps_mode', 'tracking')
            if gps_mode == 'cold_start':
                total_power += self.power_coeffs['gps']['cold_start']
            else:
                total_power += self.power_coeffs['gps']['tracking']
        
        # Camera
        camera_mode = usage_mode.get('camera_mode', 'none')
        if camera_mode == 'preview':
            total_power += self.power_coeffs['camera']['preview']
        elif camera_mode == 'photo':
            total_power += self.power_coeffs['camera']['photo']
            
            if usage_mode.get('flash_on', False):
                total_power += self.power_coeffs['camera']['flash']
        elif camera_mode == 'video_1080p':
            total_power += self.power_coeffs['camera']['video_1080p']
        elif camera_mode == 'video_4k':
            total_power += self.power_coeffs['camera']['video_4k']
        
        # Audio
        audio_output = usage_mode.get('audio_output', 'none')
        if audio_output == 'speaker':
            total_power += self.power_coeffs['audio']['speaker']
        elif audio_output == 'headphone':
            total_power += self.power_coeffs['audio']['headphone']
        
        if audio_output != 'none':
            volume = usage_mode.get('volume', 50)
            total_power += self.power_coeffs['audio']['volume_factor'] * volume
        
        # Other components
        if usage_mode.get('bluetooth_on', False):
            total_power += self.power_coeffs['bluetooth']['active']
        
        if usage_mode.get('sensors_active', False):
            total_power += self.power_coeffs['sensors']['active']
        
        if usage_mode.get('vibration_on', False):
            total_power += self.power_coeffs['vibration']['active']
        
        # Memory and storage
        if usage_mode.get('memory_active', False):
            total_power += self.power_coeffs['memory']['active']
        
        if usage_mode.get('storage_active', False):
            total_power += self.power_coeffs['storage']['active']
        
        return total_power / 1000.0  # Convert to Watts

# ==================== REAL-WORLD USAGE SCENARIOS ====================
def define_real_usage_scenarios(power_model):
    """Define realistic usage scenarios with corrected parameters"""
    
    scenarios = {
        'Standby': {
            'screen_on': False,
            'cpu_mode': 'idle',
            'cpu_util': 2,  # Very low CPU utilization
            'cellular_type': '4g',
            'cellular_active': False,
            'wifi_on': False,
            'gps_on': False,
            'camera_mode': 'none',
            'audio_output': 'none',
            'bluetooth_on': False,
            'sensors_active': False,
            'vibration_on': False
        },
        
        'Voice Call': {
            'screen_on': False,
            'cpu_mode': 'light',
            'cpu_util': 15,
            'cellular_type': '4g',
            'cellular_active': True,
            'audio_output': 'speaker',
            'volume': 60,
            'vibration_on': False
        },
        
        'iMessage Chat': {
            'screen_on': True,
            'screen_brightness': 50,
            'cpu_mode': 'light',
            'cpu_util': 20,
            'cellular_type': '4g',
            'cellular_active': True,
            'upload_rate': 0.05,
            'download_rate': 0.1,
            'audio_output': 'none',
            'vibration_on': False
        },
        
        'Web Browsing': {
            'screen_on': True,
            'screen_brightness': 60,
            'cpu_mode': 'medium',
            'cpu_util': 35,
            'wifi_on': True,
            'wifi_active': True,
            'wifi_download': 0.3,
            'wifi_upload': 0.05,
            'memory_active': True,
            'storage_active': False
        },
        
        'Video Streaming (1080p)': {
            'screen_on': True,
            'screen_brightness': 70,
            'cpu_mode': 'medium',
            'cpu_util': 45,
            'gpu_active': True,
            'gpu_mode': 'light',
            'gpu_util': 25,
            'wifi_on': True,
            'wifi_active': True,
            'wifi_download': 1.5,  # Video streaming
            'audio_output': 'headphone',
            'volume': 50,
            'memory_active': True
        },
        
        'Video Streaming (4K)': {
            'screen_on': True,
            'screen_brightness': 80,
            'cpu_mode': 'heavy',
            'cpu_util': 60,
            'gpu_active': True,
            'gpu_mode': 'heavy',
            'gpu_util': 50,
            'wifi_on': True,
            'wifi_active': True,
            'wifi_download': 5.0,  # 4K video stream
            'audio_output': 'headphone',
            'volume': 60,
            'memory_active': True,
            'storage_active': False
        },
        
        'Photo Capture': {
            'screen_on': True,
            'screen_brightness': 100,
            'cpu_mode': 'medium',
            'cpu_util': 50,
            'gpu_active': True,
            'gpu_mode': 'medium',
            'gpu_util': 30,
            'camera_mode': 'photo',
            'flash_on': False,
            'cellular_type': '4g',
            'cellular_active': False,
            'memory_active': True,
            'storage_active': True
        },
        
        'Video Recording (1080p)': {
            'screen_on': True,
            'screen_brightness': 100,
            'cpu_mode': 'heavy',
            'cpu_util': 70,
            'gpu_active': True,
            'gpu_mode': 'heavy',
            'gpu_util': 60,
            'camera_mode': 'video_1080p',
            'cellular_type': 'none',
            'cellular_active': False,
            'audio_output': 'speaker',
            'volume': 30,
            'memory_active': True,
            'storage_active': True
        },
        
        'Video Recording (4K)': {
            'screen_on': True,
            'screen_brightness': 100,
            'cpu_mode': 'heavy',
            'cpu_util': 80,
            'gpu_active': True,
            'gpu_mode': 'heavy',
            'gpu_util': 70,
            'camera_mode': 'video_4k',
            'cellular_type': 'none',
            'cellular_active': False,
            'audio_output': 'speaker',
            'volume': 10,
            'memory_active': True,
            'storage_active': True
        },
        
        'Light Gaming': {
            'screen_on': True,
            'screen_brightness': 70,
            'cpu_mode': 'medium',
            'cpu_util': 50,
            'gpu_active': True,
            'gpu_mode': 'light',
            'gpu_util': 40,
            'wifi_on': True,
            'wifi_active': True,
            'wifi_download': 0.3,
            'audio_output': 'headphone',
            'volume': 50,
            'vibration_on': False,
            'memory_active': True,
            'sensors_active': True
        },
        
        'Heavy Gaming': {
            'screen_on': True,
            'screen_brightness': 90,
            'cpu_mode': 'heavy',
            'cpu_util': 75,
            'gpu_active': True,
            'gpu_mode': 'heavy',
            'gpu_util': 65,
            'cellular_type': 'none',
            'cellular_active': False,
            'wifi_on': True,
            'wifi_active': True,
            'wifi_download': 0.8,
            'wifi_upload': 0.4,
            'audio_output': 'headphone',
            'volume': 70,
            'vibration_on': True,
            'memory_active': True,
            'storage_active': True,
            'sensors_active': True,
            'bluetooth_on': False  # Most mobile games don't use Bluetooth controllers
        },
        
        'Navigation': {
            'screen_on': True,
            'screen_brightness': 100,
            'cpu_mode': 'medium',
            'cpu_util': 45,
            'gpu_active': True,
            'gpu_mode': 'light',
            'gpu_util': 25,
            'cellular_type': '4g',
            'cellular_active': True,
            'download_rate': 0.2,
            'gps_on': True,
            'gps_mode': 'tracking',
            'audio_output': 'speaker',
            'volume': 60,
            'memory_active': True
        },
        
        'Video Conferencing': {
            'screen_on': True,
            'screen_brightness': 70,
            'cpu_mode': 'heavy',
            'cpu_util': 65,
            'gpu_active': True,
            'gpu_mode': 'medium',
            'gpu_util': 35,
            'camera_mode': 'video_1080p',
            'wifi_on': True,
            'wifi_active': True,
            'wifi_download': 1.5,
            'wifi_upload': 1.5,
            'audio_output': 'speaker',
            'volume': 50,
            'memory_active': True
        }
    }
    
    # Calculate power consumption for each scenario
    powers = {}
    for name, config in scenarios.items():
        powers[name] = power_model.calculate_power(config)
    
    return scenarios, powers

# ==================== BATTERY MODEL SOLVER ====================
class BatterySolver:
    def __init__(self, battery_params):
        self.params = battery_params
    
    def _ode_system(self, t, y, P):
        """
        Define ODE system: dy/dt = f(t, y)
        y = [SOC, Vp1, Vp2]
        """
        SOC, Vp1, Vp2 = y
        params = self.params
        
        # Calculate open-circuit voltage
        V_oc = params.V_oc(SOC)
        
        # Calculate terminal voltage (solve quadratic equation)
        # V_t^2 - (V_oc - Vp1 - Vp2)*V_t + P*R0/eta_conv = 0
        a = 1.0
        b = -(V_oc - Vp1 - Vp2)
        c = (P * params.R0) / params.eta_conv
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            # Cannot provide required power, return stop signal
            return [0, 0, 0]
        
        # Take positive root (high voltage solution)
        V_t = (-b + np.sqrt(discriminant)) / (2*a)
        
        # Calculate current
        I = P / (params.eta_conv * V_t) if V_t > 0 else 0
        
        # ODE equations
        dSOC_dt = -params.eta * I / (params.Cn * 3600 * params.SOH)  # Convert to per second
        dVp1_dt = I / params.C1 - Vp1 / (params.R1 * params.C1)
        dVp2_dt = I / params.C2 - Vp2 / (params.R2 * params.C2)
        
        return [dSOC_dt, dVp1_dt, dVp2_dt]
    
    def solve(self, P, initial_SOC=1.0, t_end=24*3600, method='RK45'):
        """
        Solve battery discharge process
        P: discharge power (W), can be constant or function P(t)
        t_end: simulation end time (seconds)
        """
        # Initial conditions
        y0 = [initial_SOC, 0.0, 0.0]  # SOC, Vp1, Vp2
        
        # Event function: stop when SOC <= 0
        def event(t, y):
            return y[0]
        event.terminal = True
        event.direction = -1
        
        # If P is constant, convert to function
        if np.isscalar(P):
            P_func = lambda t: P
        else:
            P_func = P
        
        # Define wrapper function
        def ode_wrapper(t, y):
            return self._ode_system(t, y, P_func(t))
        
        # Use adaptive step size solver
        sol = solve_ivp(
            ode_wrapper,
            [0, t_end],
            y0,
            method=method,
            events=[event],
            rtol=1e-6,
            atol=1e-9,
            max_step=10.0  # Maximum step size of 10 seconds
        )
        
        return sol

# ==================== SCIENTIFIC COLOR PALETTE ====================
class ScientificColorPalette:
    """Scientific color palette for publication-quality figures"""
    
    @staticmethod
    def get_viridis_colors(n):
        """Get n colors from the viridis colormap"""
        return plt.cm.viridis(np.linspace(0.2, 0.8, n))
    
    @staticmethod
    def get_plasma_colors(n):
        """Get n colors from the plasma colormap"""
        return plt.cm.plasma(np.linspace(0.2, 0.8, n))
    
    @staticmethod
    def get_inferno_colors(n):
        """Get n colors from the inferno colormap"""
        return plt.cm.inferno(np.linspace(0.2, 0.8, n))
    
    @staticmethod
    def get_cividis_colors(n):
        """Get n colors from the cividis colormap (colorblind-friendly)"""
        return plt.cm.cividis(np.linspace(0.2, 0.8, n))
    
    @staticmethod
    def get_tab10_colors(n):
        """Get n colors from the tab10 colormap"""
        return plt.cm.tab10(np.linspace(0, 1, n))[:n]

# ==================== REALISTIC DAILY USAGE SCENARIO ====================
def realistic_daily_usage(t):
    """
    Simulate realistic daily usage scenario with corrected power levels
    Returns time-varying power consumption (W)
    t: time (seconds)
    """
    t_hours = t / 3600.0  # Convert to hours
    
    # Typical daily time allocation with realistic power levels
    if t_hours < 1:  # 0-1 AM: Standby
        return 0.15  # Reduced standby power
    
    elif t_hours < 7:  # 1-7 AM: Sleep, deep standby
        return 0.12  # Very low power during sleep
    
    elif t_hours < 8:  # 7-8 AM: Morning, light usage
        hour_fraction = t_hours - 7
        if hour_fraction < 0.25:
            return 0.6  # Checking notifications
        elif hour_fraction < 0.5:
            return 1.2  # iMessage chat
        else:
            return 1.5  # Browsing news
    
    elif t_hours < 9:  # 8-9 AM: Commute
        return 2.8  # Navigation + music
    
    elif t_hours < 12:  # 9 AM-12 PM: Work hours
        hour_fraction = t_hours - 9
        mod = hour_fraction % 1.0
        
        if mod < 0.8:  # Mostly standby or light usage
            return 0.8 + 0.4 * np.sin(2*np.pi*hour_fraction/2)
        else:  # Occasional usage
            return 2.0  # Checking messages, emails
    
    elif t_hours < 13:  # 12-1 PM: Lunch break
        hour_fraction = t_hours - 12
        if hour_fraction < 0.3:
            return 3.2  # Short videos
        elif hour_fraction < 0.6:
            return 3.8  # Light gaming
        else:
            return 2.5  # Social chatting
    
    elif t_hours < 17:  # 1-5 PM: Afternoon work
        hour_fraction = t_hours - 13
        mod = hour_fraction % 1.0
        
        if mod < 0.7:  # Light usage
            return 1.0 + 0.3 * np.sin(2*np.pi*hour_fraction/1.5)
        elif mod < 0.8:  # Video conference
            return 3.8
        else:  # Break
            return 1.8
    
    elif t_hours < 18:  # 5-6 PM: Evening commute
        return 2.8  # Navigation + music
    
    elif t_hours < 19:  # 6-7 PM: Dinner time
        return 2.2  # Photo taking, social
    
    elif t_hours < 22:  # 7-10 PM: Evening entertainment
        hour_fraction = t_hours - 19
        
        if hour_fraction < 0.5:
            return 4.2  # Heavy gaming
        elif hour_fraction < 1.5:
            return 3.5  # 4K video streaming
        elif hour_fraction < 2.0:
            return 2.8  # Social chatting
        else:
            return 4.0  # More gaming
    
    elif t_hours < 23:  # 10-11 PM: Pre-sleep usage
        return 2.0  # Browsing social media
    
    else:  # 11 PM-12 AM: Standby
        return 0.2  # Low power standby

# ==================== MAIN PROGRAM ====================
def main():
    # Set scientific plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Initialize parameters
    battery_params = BatteryParameters()
    power_model = PowerModel()
    solver = BatterySolver(battery_params)
    color_palette = ScientificColorPalette()
    
    # Define real usage scenarios and calculate power consumption
    scenarios, powers = define_real_usage_scenarios(power_model)
    
    print("=== REAL-WORLD USAGE SCENARIO POWER ANALYSIS ===")
    print(f"{'Scenario':<25} {'Power(W)':<12} {'Theoretical Usage Time(h)':<25}")
    print("-" * 65)
    
    # Battery total energy (Wh)
    battery_energy = battery_params.Cn * 3.7 * battery_params.SOH  # Approximately 11.1Wh
    
    for name in sorted(powers.keys(), key=lambda x: powers[x], reverse=True):
        power = powers[name]
        theoretical_time = battery_energy / (power / battery_params.eta_conv)
        print(f"{name:<25} {power:<12.2f} {theoretical_time:<25.2f}")
    
    # 1. Plot SOC-t curves for most power-intensive scenarios
    print("\n=== SIMULATING DISCHARGE CURVES FOR TYPICAL SCENARIOS ===")
    
    # Select most typical scenarios
    typical_scenarios = ['Standby', 'Web Browsing', 'Video Streaming (4K)', 
                         'Heavy Gaming', 'Video Recording (4K)']
    
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    
    # Use scientific color palette
    colors = color_palette.get_viridis_colors(len(typical_scenarios))
    
    for idx, scenario_name in enumerate(typical_scenarios):
        power = powers[scenario_name]
        print(f"Simulating {scenario_name} (Power: {power:.2f}W)...")
        
        # Solve battery discharge
        sol = solver.solve(power, initial_SOC=1.0, t_end=15*3600)  # Extended simulation time
        
        if sol.success and len(sol.t) > 0:
            # Extract results
            time_hours = sol.t / 3600.0  # Convert to hours
            SOC = sol.y[0] * 100  # Convert to percentage
            
            # Find time when SOC reaches 0
            discharge_time = sol.t[-1] / 3600.0
            
            # Plot curve with scientific styling
            ax1.plot(time_hours, SOC, 
                    label=f'{scenario_name} ({power:.2f}W, {discharge_time:.1f}h)',
                    color=colors[idx],
                    linewidth=2.5,
                    alpha=0.9)
            
            # Add marker at battery depletion
            if len(time_hours) > 0 and SOC[-1] <= 0:
                ax1.plot(time_hours[-1], SOC[-1], 'o', 
                        color=colors[idx], markersize=8)
    
    # Scientific plot styling
    ax1.set_xlabel('Time (hours)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('State of Charge (%)', fontsize=16, fontweight='bold')
    ax1.set_title('Battery Discharge Characteristics for Different Usage Scenarios', 
                 fontsize=18, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax1.set_ylim(-5, 105)
    ax1.set_xlim(0, 15)
    
    # Add grid lines for important SOC levels
    ax1.axhline(y=20, color='orange', linestyle=':', alpha=0.7, linewidth=1)
    ax1.axhline(y=10, color='red', linestyle=':', alpha=0.7, linewidth=1)
    
    # Improve tick labels
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig('battery_discharge_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Simulate realistic daily usage scenario
    print("\n=== SIMULATING REALISTIC DAILY USAGE ===")
    
    sol_daily = solver.solve(realistic_daily_usage, initial_SOC=1.0, t_end=48*3600)
    
    if sol_daily.success and len(sol_daily.t) > 0:
        # Calculate battery depletion time
        depletion_time = sol_daily.t[-1] / 3600.0
        
        print(f"Battery depletion time: {depletion_time:.2f} hours")
        print(f"Equivalent to {int(depletion_time//24)} days and {depletion_time%24:.1f} hours")
        
        # Create detailed analysis plot with scientific styling
        fig2, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
        
        # Subplot 1: SOC-t curve
        time_hours = sol_daily.t / 3600.0
        SOC = sol_daily.y[0] * 100
        
        # Use plasma colormap for main curve
        axes[0].plot(time_hours, SOC, 'b-', linewidth=2.5, alpha=0.9)
        axes[0].fill_between(time_hours, 0, SOC, alpha=0.2, color='blue')
        
        # Add reference lines with scientific styling
        axes[0].axhline(y=20, color='orange', linestyle='--', alpha=0.8, 
                       linewidth=1.5, label='Low Battery (20%)')
        axes[0].axhline(y=10, color='red', linestyle='--', alpha=0.8, 
                       linewidth=1.5, label='Critical Battery (10%)')
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        if depletion_time < 24:
            axes[0].axvline(x=depletion_time, color='darkred', linestyle='--', 
                           alpha=0.8, linewidth=1.5)
            axes[0].text(depletion_time, 50, f'Depletion\n{depletion_time:.1f}h', 
                        rotation=90, verticalalignment='center',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', 
                                alpha=0.9, edgecolor='darkred'))
        
        # Scientific styling for subplot 1
        axes[0].set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('SOC (%)', fontsize=14, fontweight='bold')
        axes[0].set_title('Daily Battery Discharge Profile', 
                         fontsize=16, fontweight='bold', pad=15)
        axes[0].grid(True, alpha=0.3, linestyle=':')
        axes[0].set_ylim(-5, 105)
        axes[0].set_xlim(0, min(24, depletion_time + 2))
        axes[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        
        # Subplot 2: Power consumption over time
        power_values = []
        for t in time_hours:
            power_values.append(realistic_daily_usage(t * 3600))
        
        # Use inferno colormap for power curve
        axes[1].plot(time_hours, power_values, '-', color='darkorange', 
                    linewidth=2, alpha=0.9)
        axes[1].fill_between(time_hours, 0, power_values, alpha=0.3, 
                            color='darkorange')
        
        # Scientific styling for subplot 2
        axes[1].set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Power (W)', fontsize=14, fontweight='bold')
        axes[1].set_title('Daily Power Consumption Pattern', 
                         fontsize=16, fontweight='bold', pad=15)
        axes[1].grid(True, alpha=0.3, linestyle=':')
        axes[1].set_xlim(0, min(24, depletion_time + 2))
        axes[1].tick_params(axis='both', which='major', labelsize=12)
        
        # Add usage period annotations with scientific colors
        usage_labels = [
            (0, 7, "Sleep", "#4B0082"),        # Indigo
            (7, 9, "Morning", "#008080"),       # Teal
            (9, 12, "Work AM", "#228B22"),      # Forest Green
            (12, 13, "Lunch", "#FF8C00"),       # Dark Orange
            (13, 17, "Work PM", "#228B22"),     # Forest Green
            (17, 19, "Evening", "#008080"),     # Teal
            (19, 22, "Entertainment", "#8B0000"), # Dark Red
            (22, 24, "Pre-sleep", "#4B0082")    # Indigo
        ]
        
        for start, end, label, color in usage_labels:
            if depletion_time > start:
                end_actual = min(end, depletion_time)
                axes[1].axvspan(start, end_actual, alpha=0.15, color=color)
                axes[1].text((start + end_actual)/2, max(power_values)/2, 
                            label, ha='center', va='center', fontsize=10,
                            fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', 
                                    alpha=0.9, edgecolor=color))
        
        # Subplot 3: Discharge rate (dSOC/dt)
        dSOC_dt = np.gradient(SOC, time_hours)  # % per hour
        
        # Use cividis colormap for discharge rate (colorblind friendly)
        axes[2].plot(time_hours, -dSOC_dt, '-', color='purple', 
                    linewidth=2, alpha=0.9)
        
        # Scientific styling for subplot 3
        axes[2].set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Discharge Rate (%/h)', fontsize=14, fontweight='bold')
        axes[2].set_title('Battery Discharge Rate', 
                         fontsize=16, fontweight='bold', pad=15)
        axes[2].grid(True, alpha=0.3, linestyle=':')
        axes[2].set_xlim(0, min(24, depletion_time + 2))
        axes[2].tick_params(axis='both', which='major', labelsize=12)
        
        # Mark high discharge rate regions
        high_discharge_threshold = 15  # %/h
        high_discharge_mask = -dSOC_dt > high_discharge_threshold
        if np.any(high_discharge_mask):
            axes[2].fill_between(time_hours, 0, high_discharge_threshold, 
                                where=high_discharge_mask, alpha=0.2, color='red')
            axes[2].text(np.mean(time_hours[high_discharge_mask]), 
                        high_discharge_threshold*1.3,
                        "High Power Activities", ha='center', fontsize=11,
                        fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', 
                                alpha=0.9, edgecolor='red'))
        
        plt.tight_layout()
        plt.savefig('realistic_daily_usage.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Analyze impact of different scenarios on battery life
        print("\n=== SCENARIO POWER CONSUMPTION COMPARISON ===")
        
        # Create horizontal bar chart with scientific styling
        fig3, ax3 = plt.subplots(figsize=(14, 8))
        
        # Sort by power consumption
        sorted_scenarios = sorted(powers.items(), key=lambda x: x[1], reverse=True)
        scenario_names = [name for name, _ in sorted_scenarios]
        scenario_powers = [power for _, power in sorted_scenarios]
        
        # Calculate theoretical usage time
        theoretical_times = [battery_energy / (p / battery_params.eta_conv) for p in scenario_powers]
        
        # Use viridis colormap for bars
        colors_bar = color_palette.get_viridis_colors(len(scenario_names))
        bars = ax3.barh(scenario_names, scenario_powers, color=colors_bar, alpha=0.9)
        
        # Add value labels with scientific formatting
        for i, (bar, power, time) in enumerate(zip(bars, scenario_powers, theoretical_times)):
            ax3.text(power + 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{power:.2f}W ({time:.1f}h)', 
                   va='center', ha='left', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', 
                           alpha=0.9, pad=0.2))
        
        # Scientific styling for bar chart
        ax3.set_xlabel('Power Consumption (W)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Usage Scenario', fontsize=14, fontweight='bold')
        ax3.set_title('Comparative Power Consumption of Smartphone Usage Scenarios', 
                     fontsize=16, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3, axis='x', linestyle=':')
        ax3.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.savefig('scenario_power_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Energy-saving recommendations
        print("\n=== ENERGY-SAVING RECOMMENDATIONS ===")
        print("Based on model analysis, the following measures can significantly extend battery life:")
        print("1. Avoid continuous 4K video recording: This consumes ~3.5W, limiting usage to ~3 hours")
        print("2. Reduce screen brightness during gaming: Lowering from 90% to 70% saves ~20% screen power")
        print("3. Use WiFi instead of 5G: 5G consumes ~40% more power than WiFi for data transfer")
        print("4. Close unnecessary background applications: Reduces CPU utilization and memory activity")
        print("5. Use phone in areas with good signal: Weak signals increase transmission power by up to 300%")
        print("6. Use dark mode (AMOLED screens): Can reduce screen power by up to 60% for dark content")
        print("7. Reduce auto-lock time to 30 seconds: Minimizes unnecessary screen-on time")
        print("8. Disable GPS when not in use: Continuous GPS tracking consumes ~120mW")
        print("9. Use airplane mode in low-signal areas: Prevents power-intensive network searching")
        print("10. Limit background app refresh: Reduces CPU and network activity")
        
        # 5. Calculate energy-saving effects
        print("\n=== ENERGY-SAVING EFFECT ESTIMATION ===")
        
        # Compare power consumption under different settings for heavy gaming
        base_config = scenarios['Heavy Gaming'].copy()
        power_base = power_model.calculate_power(base_config)
        
        # Energy-saving configuration
        energy_saving_config = base_config.copy()
        energy_saving_config['screen_brightness'] = 70  # Lower brightness
        energy_saving_config['cpu_util'] = 65  # Slightly lower CPU utilization
        energy_saving_config['gpu_util'] = 55  # Optimized graphics settings
        energy_saving_config['volume'] = 60  # Lower volume
        energy_saving_config['wifi_download'] = 0.5  # Lower network usage
        power_saving = power_model.calculate_power(energy_saving_config)
        
        improvement = (power_base - power_saving) / power_base * 100
        
        print(f"Heavy Gaming Scenario:")
        print(f"  Default settings: {power_base:.2f}W, Theoretical usage: {battery_energy/(power_base/battery_params.eta_conv):.1f}h")
        print(f"  Energy-saving settings: {power_saving:.2f}W, Theoretical usage: {battery_energy/(power_saving/battery_params.eta_conv):.1f}h")
        print(f"  Battery life improvement: {improvement:.1f}%")
        
        # 6. Additional analysis: Battery aging effects
        print("\n=== BATTERY AGING EFFECTS ===")
        
        # Simulate battery aging (reduced capacity)
        aging_levels = [1.0, 0.9, 0.8, 0.7]  # SOH values
        discharge_times_aging = []
        
        for soh in aging_levels:
            aged_params = BatteryParameters()
            aged_params.SOH = soh
            
            aged_solver = BatterySolver(aged_params)
            
            # Solve for typical scenario
            power = powers['Web Browsing']
            sol = aged_solver.solve(power, initial_SOC=1.0, t_end=10*3600)
            
            discharge_time = sol.t[-1] / 3600.0 if sol.success else 0
            discharge_times_aging.append((soh, discharge_time))
            
            print(f"  SOH = {soh*100:.0f}%: Discharge time = {discharge_time:.2f}h "
                  f"(Reduction: {(1-discharge_time/discharge_times_aging[0][1])*100:.1f}%)")
        
        # Plot battery aging effects
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        soh_vals = [x[0]*100 for x in discharge_times_aging]
        time_vals = [x[1] for x in discharge_times_aging]
        
        # Use plasma colormap for aging plot
        ax4.plot(soh_vals, time_vals, 's-', linewidth=2.5, markersize=10, 
                color=plt.cm.plasma(0.6), alpha=0.9)
        
        # Scientific styling
        ax4.set_xlabel('State of Health (%)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Discharge Time (hours)', fontsize=14, fontweight='bold')
        ax4.set_title('Battery Aging Impact on Usage Time (Web Browsing)', 
                     fontsize=16, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle=':')
        ax4.tick_params(axis='both', which='major', labelsize=12)
        
        # Add annotations
        for i, (soh, time) in enumerate(zip(soh_vals, time_vals)):
            if i > 0:
                reduction = (1 - time/time_vals[0]) * 100
                ax4.annotate(f'-{reduction:.1f}%', 
                           xy=(soh, time), xytext=(0, 10),
                           textcoords='offset points', ha='center',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', 
                                   alpha=0.9, pad=0.2))
        
        plt.tight_layout()
        plt.savefig('battery_aging_effects.png', dpi=300, bbox_inches='tight')
        plt.show()

# ==================== EXECUTE MAIN PROGRAM ====================
if __name__ == "__main__":
    print("Starting simulation of smartphone battery discharge behavior...")
    print("=" * 70)
    main()
    print("\nSimulation complete! All charts saved as PNG files.")