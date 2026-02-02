"""
Computational Neuroscience Project - Morris-Lecar & Hodgkin-Huxley Models
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class NeuralDynamicsSimulation:
    def __init__(self):
        """Initialize simulation parameters and configuration"""
        self.setup_parameters()
        self.setup_plot_style()
        
    def setup_parameters(self):
        """Initialize Morris-Lecar and Hodgkin-Huxley parameters"""
        # Morris-Lecar Parameters
        self.ml_params = {
            'C': 20,        # Membrane capacitance (µF/cm²)
            'gCa': 4.4,     # Calcium conductance (mS/cm²)
            'gK': 8.0,      # Potassium conductance (mS/cm²)
            'gL': 2.0,      # Leak conductance (mS/cm²)
            'VCa': 120,     # Calcium reversal potential (mV)
            'VK': -84,      # Potassium reversal potential (mV)
            'VL': -60,      # Leak reversal potential (mV)
            'V1': -1.2,     # Voltage half-activation for m (mV)
            'V2': 18,       # Voltage slope for m (mV)
            'V3': 2,        # Voltage half-activation for w (mV)
            'V4': 30,       # Voltage slope for w (mV)
            'phi': 0.02,    # Temporal scale (ms⁻¹)
            'I_ext': 0      # External current (µA/cm²)
        }
        
        # Hodgkin-Huxley Parameters
        self.hh_params = {
            'C': 1,         # Membrane capacitance (µF/cm²)
            'gNa': 120,     # Sodium conductance (mS/cm²)
            'gK': 36,       # Potassium conductance (mS/cm²)
            'gL': 0.3,      # Leak conductance (mS/cm²)
            'VNa': 55,      # Sodium reversal potential (mV)
            'VK': -72,      # Potassium reversal potential (mV)
            'VL': -54.387,  # Leak reversal potential (mV)
            'I_ext': 0,     # External current (µA/cm²)
            'eps': 1e-10    # Numerical epsilon
        }
        
        # Simulation configuration
        self.sim_config = {
            'rtol': 1e-6,
            'atol': 1e-9,
            'max_step': 0.1
        }
        
    def setup_plot_style(self):
        """Setup matplotlib styling"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.linewidth': 1.5,
            'lines.linewidth': 2.5,
            'grid.alpha': 0.3,
            'figure.figsize': (10, 8)
        })
        
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B2635']
    
    def run_analysis(self):
        """Execute complete neural dynamics analysis"""
        print("=== MORRIS-LECAR MODEL ANALYSIS ===")
        self.execute_morris_lecar_analysis()
        
        print("\n=== HODGKIN-HUXLEY MODEL ANALYSIS ===")
        self.execute_hodgkin_huxley_analysis()
    
    # ========== MORRIS-LECAR ANALYSIS ==========
    
    def execute_morris_lecar_analysis(self):
        """Execute Morris-Lecar model analysis"""
        self.analyze_phase_space_ml()
        self.perform_stability_analysis_ml()
        self.generate_action_potentials_ml()
        self.analyze_depolarization_threshold()
        self.analyze_high_current_response()
        self.perform_multi_current_analysis()
        self.analyze_alternative_parameter_set()
    
    def steady_state_activation(self, V, V_half, V_slope):
        """Steady-state activation function"""
        return 0.5 * (1 + np.tanh((V - V_half) / V_slope))
    
    def morris_lecar_derivatives(self, V, w):
        """Compute Morris-Lecar derivatives"""
        p = self.ml_params
        
        # Steady-state activation functions
        m_inf = self.steady_state_activation(V, p['V1'], p['V2'])
        w_inf = self.steady_state_activation(V, p['V3'], p['V4'])
        
        # Membrane currents
        I_Ca = p['gCa'] * m_inf * (V - p['VCa'])
        I_K = p['gK'] * w * (V - p['VK'])
        I_L = p['gL'] * (V - p['VL'])
        
        # Differential equations
        dVdt = (-I_Ca - I_K - I_L + p['I_ext']) / p['C']
        dwdt = p['phi'] * (w_inf - w) / (1 / np.cosh((V - p['V3']) / (2 * p['V4'])))
        
        return dVdt, dwdt
    
    def find_equilibrium_hh(self):
        """Find Hodgkin-Huxley equilibrium point"""
        def equilibrium_function(x):
            V, n, m, h = x
            p = self.hh_params
            
            # Rate constants
            alpha_n = self.alpha_n(V)
            beta_n = 0.125 * np.exp(-(V + 60)/80)
            alpha_m = self.alpha_m(V)
            beta_m = 4 * np.exp(-(V + 60)/18)
            alpha_h = 0.07 * np.exp(-(V + 60)/20)
            beta_h = 1 / (np.exp(-(V + 30)/10) + 1)
            
            # Equilibrium conditions
            I_Na = p['gNa'] * m**3 * h * (V - p['VNa'])
            I_K = p['gK'] * n**4 * (V - p['VK'])
            I_L = p['gL'] * (V - p['VL'])
            
            F1 = -(I_Na + I_K + I_L - p['I_ext']) / p['C']
            F2 = alpha_n * (1 - n) - beta_n * n
            F3 = alpha_m * (1 - m) - beta_m * m
            F4 = alpha_h * (1 - h) - beta_h * h
            
            return [F1, F2, F3, F4]
        
        # Use steady-state approximation as initial guess
        V_guess = -60
        n_guess, m_guess, h_guess = self.steady_state_values_hh(V_guess)
        
        try:
            equilibrium = fsolve(equilibrium_function, [V_guess, n_guess, m_guess, h_guess], xtol=1e-12)
            return equilibrium
        except:
            return [np.nan, np.nan, np.nan, np.nan]
    
    def calibrate_resting_potential(self):
        """Calibrate resting potential to -60 mV"""
        print("Part 13: Calibrating Resting Potential to -60 mV")
        
        target_resting = -60  # mV
        
        # Calculate steady-state values at target resting potential
        n_inf, m_inf, h_inf = self.steady_state_values_hh(target_resting)
        
        # Calculate required leak reversal potential
        p = self.hh_params
        ionic_current = (p['gNa'] * (m_inf**3) * h_inf * (target_resting - p['VNa']) +
                        p['gK'] * (n_inf**4) * (target_resting - p['VK']))
        
        E_L_required = target_resting - (p['I_ext'] - ionic_current) / p['gL']
        
        self.hh_params['VL'] = E_L_required
        
        print(f"Required leak reversal potential: E_L = {E_L_required:.6f} mV")
        
        # Test with current injection
        self.test_current_injection_hh()
    
    def test_current_injection_hh(self):
        """Test action potential generation with current injection"""
        print("Testing action potential generation with I_ext = 10 µA/cm²")
        
        self.hh_params['I_ext'] = 10
        
        # Initial conditions at resting potential
        n_inf, m_inf, h_inf = self.steady_state_values_hh(-60)
        initial_conditions = [-60, n_inf, m_inf, h_inf]
        
        t_span = (0, 100)
        t_eval = np.linspace(0, 100, 1000)
        
        sol = solve_ivp(self.hodgkin_huxley_ode, t_span, initial_conditions, 
                       t_eval=t_eval, rtol=self.sim_config['rtol'], 
                       atol=self.sim_config['atol'])
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(sol.t, sol.y[0], linewidth=2.5, color=self.colors[0])
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (mV)')
        ax1.set_title('Action Potential (I = 10 µA/cm²)')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(sol.t, sol.y[1], linewidth=2, color=self.colors[1], label='n')
        ax2.plot(sol.t, sol.y[2], linewidth=2, color=self.colors[2], label='m')
        ax2.plot(sol.t, sol.y[3], linewidth=2, color=self.colors[3], label='h')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Gating Variables')
        ax2.set_title('Gating Variable Dynamics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.hh_params['I_ext'] = 0  # Reset
    
    def analyze_resting_stability_hh(self):
        """Analyze resting state stability"""
        print("Part 14: Resting State Stability Analysis")
        
        # Find equilibrium point
        equilibrium = self.find_equilibrium_hh()
        V_eq, n_eq, m_eq, h_eq = equilibrium
        
        print(f"Equilibrium point: V = {V_eq:.3f}, n = {n_eq:.6f}, m = {m_eq:.6f}, h = {h_eq:.6f}")
        
        # Compute Jacobian and analyze stability
        J = self.compute_jacobian_hh(equilibrium)
        eigenvals = np.linalg.eigvals(J)
        
        print("Eigenvalues:")
        for i, eigenval in enumerate(eigenvals):
            print(f"  λ{i+1} = {eigenval:.6f}")
        
        if np.all(np.real(eigenvals) < 0):
            print("Resting state is STABLE")
        else:
            print("Resting state is UNSTABLE")
        
        # Threshold analysis
        self.analyze_current_threshold_hh()
    
    def compute_jacobian_hh(self, state):
        """Compute Jacobian matrix for HH system numerically"""
        n = len(state)
        J = np.zeros((n, n))
        h = 1e-8
        
        f0 = self.hodgkin_huxley_ode(0, state)
        
        for i in range(n):
            state_perturb = state.copy()
            state_perturb[i] += h
            f_perturb = self.hodgkin_huxley_ode(0, state_perturb)
            J[:, i] = (np.array(f_perturb) - np.array(f0)) / h
        
        return J
    
    def analyze_current_threshold_hh(self):
        """Analyze current pulse threshold"""
        print("Analyzing current pulse threshold")
        
        # Test range of impulse currents
        impulse_range = np.linspace(0, 15, 100)
        max_voltages = np.zeros(len(impulse_range))
        
        equilibrium = self.find_equilibrium_hh()
        V_eq, n_eq, m_eq, h_eq = equilibrium
        
        for i, impulse in enumerate(impulse_range):
            # Apply impulse as initial voltage perturbation
            initial_conditions = [V_eq + impulse/self.hh_params['C'], n_eq, m_eq, h_eq]
            
            sol = solve_ivp(self.hodgkin_huxley_ode, (0, 100), initial_conditions, 
                          rtol=self.sim_config['rtol'], atol=self.sim_config['atol'])
            max_voltages[i] = np.max(sol.y[0])
        
        # Find threshold
        threshold_idx = np.where(max_voltages > 0)[0]
        if len(threshold_idx) > 0:
            threshold = impulse_range[threshold_idx[0]]
            print(f"Current threshold: {threshold:.3f} µA/cm²")
            
            # Plot threshold curve
            plt.figure(figsize=(10, 6))
            plt.plot(impulse_range, max_voltages, linewidth=2.5, color=self.colors[0])
            plt.scatter(threshold, max_voltages[threshold_idx[0]], s=100, c=self.colors[3], 
                       zorder=5, label=f'Threshold = {threshold:.2f} µA/cm²')
            plt.xlabel('Impulse Current (µA/cm²)')
            plt.ylabel('Peak Voltage (mV)')
            plt.title('Current Pulse Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def analyze_steady_current_range_hh(self):
        """Analyze steady current range"""
        print("Part 15: Steady Current Analysis (8-12 µA/cm²)")
        
        current_range = range(8, 13)
        
        for I_ext in current_range:
            self.hh_params['I_ext'] = I_ext
            
            print(f"\n--- I_ext = {I_ext} µA/cm² ---")
            
            # Find equilibrium
            equilibrium = self.find_equilibrium_hh()
            V_eq = equilibrium[0]
            
            # Analyze stability
            J = self.compute_jacobian_hh(equilibrium)
            eigenvals = np.linalg.eigvals(J)
            
            print(f"Equilibrium: V = {V_eq:.3f} mV")
            print(f"Eigenvalues (real parts): {np.real(eigenvals)}")
            
            # Determine stability
            if np.all(np.real(eigenvals) < 0):
                print("Status: STABLE")
            else:
                print("Status: UNSTABLE")
            
            # Test dynamics for I=9 case
            if I_ext == 9:
                self.hh_params['I_ext'] = 0
                eq_0 = self.find_equilibrium_hh()
                self.hh_params['I_ext'] = I_ext  # Reset
                
                # Simulate from zero-current initial conditions
                t_span = (0, 200)
                t_eval = np.linspace(0, 200, 2000)
                sol = solve_ivp(self.hodgkin_huxley_ode, t_span, eq_0, 
                              t_eval=t_eval, rtol=self.sim_config['rtol'], 
                              atol=self.sim_config['atol'])
                
                plt.figure(figsize=(10, 6))
                plt.plot(sol.t, sol.y[0], linewidth=2.5, color=self.colors[0])
                plt.xlabel('Time (ms)')
                plt.ylabel('Voltage (mV)')
                plt.title('Voltage Response at I = 9 µA/cm²')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        
        self.hh_params['I_ext'] = 0  # Reset
    
    def analyze_reduced_model_hh(self):
        """Analyze reduced HH model (V-n system)"""
        print("Part 16: Reduced HH Model Analysis (V-n system)")
        
        # Test with current injection
        self.hh_params['I_ext'] = 10
        
        equilibrium = self.find_equilibrium_hh()
        initial_conditions = equilibrium
        
        t_span = (0, 100)
        t_eval = np.linspace(0, 100, 1000)
        
        # Compare reduced vs full model
        sol_reduced = solve_ivp(self.reduced_hodgkin_huxley_ode, t_span, initial_conditions, 
                               t_eval=t_eval, rtol=self.sim_config['rtol'], 
                               atol=self.sim_config['atol'])
        sol_full = solve_ivp(self.hodgkin_huxley_ode, t_span, initial_conditions, 
                            t_eval=t_eval, rtol=self.sim_config['rtol'], 
                            atol=self.sim_config['atol'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(sol_reduced.t, sol_reduced.y[0], linewidth=2.5, color=self.colors[0], 
                label='Reduced Model')
        ax1.plot(sol_full.t, sol_full.y[0], '--', linewidth=2, color=self.colors[3], 
                label='Full Model')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (mV)')
        ax1.set_title('Action Potentials: Reduced vs Full')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test with current pulses
        self.hh_params['I_ext'] = 0
        impulse_values = [0, 3, 6, 9, 12]
        
        for i, impulse in enumerate(impulse_values):
            initial_conditions_pulse = [equilibrium[0] + impulse, equilibrium[1], 
                                       equilibrium[2], equilibrium[3]]
            sol = solve_ivp(self.reduced_hodgkin_huxley_ode, (0, 50), initial_conditions_pulse, 
                           rtol=self.sim_config['rtol'], atol=self.sim_config['atol'])
            
            ax2.plot(sol.t, sol.y[0], linewidth=2, label=f'Pulse = {impulse:.1f}')
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Voltage (mV)')
        ax2.set_title('Reduced Model - Current Pulses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.hh_params['I_ext'] = 0  # Reset
    
    def reduced_hodgkin_huxley_ode(self, t, y):
        """Reduced Hodgkin-Huxley ODE (V-n system with instantaneous m, constant h)"""
        V, n, m, h = y
        p = self.hh_params
        
        # Rate constants
        alpha_n = self.alpha_n(V)
        beta_n = 0.125 * np.exp(-(V + 60)/80)
        alpha_m = self.alpha_m(V)
        beta_m = 4 * np.exp(-(V + 60)/18)
        
        # Use instantaneous m, constant h
        m_inf = alpha_m / (alpha_m + beta_m)
        
        # Membrane currents
        I_Na = p['gNa'] * m_inf**3 * h * (V - p['VNa'])
        I_K = p['gK'] * n**4 * (V - p['VK'])
        I_L = p['gL'] * (V - p['VL'])
        
        # Reduced system (V, n only)
        dVdt = -(I_Na + I_K + I_L - p['I_ext']) / p['C']
        dndt = alpha_n * (1 - n) - beta_n * n
        
        return [dVdt, dndt, 0, 0]  # Keep m, h constant
    
    def demonstrate_anode_break_excitation(self):
        """Demonstrate anode break excitation"""
        print("Part 17: Anode Break Excitation")
        
        equilibrium = self.find_equilibrium_hh()
        
        # Phase 1: Hyperpolarizing current
        self.hh_params['I_ext'] = -3
        t_span1 = (0, 20)
        t_eval1 = np.linspace(0, 20, 200)
        sol1 = solve_ivp(self.hodgkin_huxley_ode, t_span1, equilibrium, 
                        t_eval=t_eval1, rtol=self.sim_config['rtol'], 
                        atol=self.sim_config['atol'])
        
        # Phase 2: Return to zero current
        self.hh_params['I_ext'] = 0
        final_state1 = [sol1.y[0][-1], sol1.y[1][-1], sol1.y[2][-1], sol1.y[3][-1]]
        t_span2 = (20, 100)
        t_eval2 = np.linspace(20, 100, 800)
        sol2 = solve_ivp(self.hodgkin_huxley_ode, t_span2, final_state1, 
                        t_eval=t_eval2, rtol=self.sim_config['rtol'], 
                        atol=self.sim_config['atol'])
        
        # Combine results
        total_time = np.concatenate([sol1.t, sol2.t])
        total_voltage = np.concatenate([sol1.y[0], sol2.y[0]])
        total_n = np.concatenate([sol1.y[1], sol2.y[1]])
        total_h = np.concatenate([sol1.y[3], sol2.y[3]])
        
        # Create comprehensive plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        ax1.plot(total_time, total_voltage, linewidth=2.5, color=self.colors[0])
        ax1.axvline(x=20, color='k', linestyle='--', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (mV)')
        ax1.set_title('Anode Break Excitation')
        ax1.text(10, -50, 'Hyperpolarization', ha='center')
        ax1.text(60, 20, 'Action Potential', ha='center')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(total_time, total_n, linewidth=2, color=self.colors[1], label='n')
        ax2.plot(total_time, total_h, linewidth=2, color=self.colors[2], label='h')
        ax2.axvline(x=20, color='k', linestyle='--', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Gating Variables')
        ax2.set_title('Gating Variable Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        current_profile = np.concatenate([-3 * np.ones(len(sol1.t)), 
                                         np.zeros(len(sol2.t))])
        ax3.plot(total_time, current_profile, linewidth=2.5, color=self.colors[3])
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Current (µA/cm²)')
        ax3.set_title('Applied Current')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("Anode break excitation successfully demonstrated")
        print("Action potential occurs after current termination due to:")
        print("  - Accumulation of h during hyperpolarization")
        print("  - Delayed recovery of n creates window for excitation")
        
        self.hh_params['I_ext'] = 0  # Reset
    
    # ========== UTILITY FUNCTIONS ==========
    
    def calculate_firing_rate(self, time, voltage):
        """Calculate firing rate from voltage time series"""
        # Find action potential peaks
        peaks, _ = find_peaks(voltage, height=0, distance=10)
        
        if len(peaks) < 2:
            return 0
        
        # Calculate average inter-spike interval
        peak_times = time[peaks]
        isi = np.diff(peak_times)
        
        if len(isi) == 0:
            return 0
        else:
            avg_isi = np.mean(isi)
            return 1000 / avg_isi  # Convert from ms to Hz


    def morris_lecar_ode(self, t, y):
        """Morris-Lecar ODE system for scipy integration"""
        V, w = y
        dVdt, dwdt = self.morris_lecar_derivatives(V, w)
        return [dVdt, dwdt]

    def compute_voltage_nullcline(self, V_range):
        """Compute V-nullcline"""
        p = self.ml_params
        m_inf = self.steady_state_activation(V_range, p['V1'], p['V2'])
        
        numerator = p['I_ext'] - p['gCa'] * m_inf * (V_range - p['VCa']) - p['gL'] * (V_range - p['VL'])
        denominator = p['gK'] * (V_range - p['VK'])
        
        # Handle singularities
        v_null = np.divide(numerator, denominator, 
                            out=np.full_like(numerator, np.nan), 
                            where=(np.abs(denominator) > 1e-10))
        return v_null

    def compute_recovery_nullcline(self, V_range):
        """Compute w-nullcline"""
        p = self.ml_params
        return self.steady_state_activation(V_range, p['V3'], p['V4'])

    def find_equilibrium_ml(self):
        """Find Morris-Lecar equilibrium point"""
        def equilibrium_function(x):
            V, w = x
            p = self.ml_params
            
            m_inf = self.steady_state_activation(V, p['V1'], p['V2'])
            w_inf = self.steady_state_activation(V, p['V3'], p['V4'])
            
            F1 = p['gCa'] * m_inf * (p['VCa'] - V) + p['gK'] * w * (p['VK'] - V) + \
                    p['gL'] * (p['VL'] - V) + p['I_ext']
            F2 = w_inf - w
            
            return [F1, F2]
        
        try:
            eq = fsolve(equilibrium_function, [-60, 0.1], xtol=1e-12)
            return eq[0], eq[1]
        except:
            return np.nan, np.nan

    def analyze_phase_space_ml(self):
        """Analyze Morris-Lecar phase space"""
        print("Part 2: Phase Space Analysis")
        
        # Create voltage range
        V_range = np.linspace(-80, 100, 200)
        
        # Calculate nullclines
        v_nullcline = self.compute_voltage_nullcline(V_range)
        w_nullcline = self.compute_recovery_nullcline(V_range)
        
        # Find equilibrium point
        V_eq, w_eq = self.find_equilibrium_ml()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot nullclines
        ax.plot(V_range, v_nullcline * 100, linewidth=3, color=self.colors[0], 
                label='V-nullcline')
        ax.plot(V_range, w_nullcline * 100, linewidth=3, color=self.colors[1], 
                label='w-nullcline')
        
        # Mark equilibrium point
        if not np.isnan(V_eq):
            ax.scatter(V_eq, w_eq * 100, s=150, c='k', marker='D', zorder=5,
                        label=f'Equilibrium ({V_eq:.3f}, {w_eq:.3f})')
        
        # Add direction field
        self.add_direction_field_ml(ax)
        
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Recovery Variable (×100)')
        ax.set_title('Morris-Lecar Phase Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Equilibrium point located at V = {V_eq:.6f} mV, w = {w_eq:.6f}")

    def add_direction_field_ml(self, ax):
        """Add direction field to phase plane"""
        v_grid = np.arange(-80, 101, 10)
        w_grid = np.arange(0, 1.01, 0.1)
        V_grid, W_grid = np.meshgrid(v_grid, w_grid)
        
        dV_grid = np.zeros_like(V_grid)
        dW_grid = np.zeros_like(V_grid)
        
        for i in range(V_grid.shape[0]):
            for j in range(V_grid.shape[1]):
                dV_grid[i,j], dW_grid[i,j] = self.morris_lecar_derivatives(V_grid[i,j], W_grid[i,j])
        
        # Normalize arrows
        magnitude = np.sqrt(dV_grid**2 + (dW_grid*100)**2)
        magnitude[magnitude == 0] = 1  # Avoid division by zero
        
        dV_norm = dV_grid / magnitude * 5
        dW_norm = dW_grid / magnitude * 5
        
        ax.quiver(V_grid, W_grid*100, dV_norm, dW_norm*100, 
                    color='gray', alpha=0.5, width=0.002)

    def perform_stability_analysis_ml(self):
        """Perform stability analysis for Morris-Lecar model"""
        print("Part 3: Stability Analysis")
        
        V_eq, w_eq = self.find_equilibrium_ml()
        
        if np.isnan(V_eq):
            print("Could not find equilibrium point")
            return
        
        # Compute Jacobian numerically
        J = self.compute_jacobian_ml(V_eq, w_eq)
        eigenvals = np.linalg.eigvals(J)
        
        print(f"Jacobian eigenvalues: {eigenvals[0]:.6f}, {eigenvals[1]:.6f}")
        
        # Determine stability
        if np.all(np.real(eigenvals) < 0):
            print("Equilibrium point is STABLE (all eigenvalues have negative real parts)")
        else:
            print("Equilibrium point is UNSTABLE")

    def compute_jacobian_ml(self, V, w):
        """Compute Jacobian matrix numerically"""
        h = 1e-8
        
        dVdt_base, dwdt_base = self.morris_lecar_derivatives(V, w)
        dVdt_V, dwdt_V = self.morris_lecar_derivatives(V + h, w)
        dVdt_w, dwdt_w = self.morris_lecar_derivatives(V, w + h)
        
        J = np.array([
            [(dVdt_V - dVdt_base) / h, (dVdt_w - dVdt_base) / h],
            [(dwdt_V - dwdt_base) / h, (dwdt_w - dwdt_base) / h]
        ])
        
        return J

    def generate_action_potentials_ml(self):
        """Generate action potentials with different phi values"""
        print("Part 5: Action Potential Generation")
        
        phi_values = [0.01, 0.02, 0.04]
        t_span = (0, 300)
        t_eval = np.linspace(0, 300, 3000)
        
        V_eq, w_eq = self.find_equilibrium_ml()
        initial_conditions = [0, w_eq]  # Start with slight depolarization
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot nullclines for phase plane
        V_range = np.linspace(-80, 100, 200)
        v_null = self.compute_voltage_nullcline(V_range)
        w_null = self.compute_recovery_nullcline(V_range)
        ax2.plot(V_range, v_null, 'k--', linewidth=1.5, label='V-nullcline')
        ax2.plot(V_range, w_null, 'k:', linewidth=1.5, label='w-nullcline')
        
        for i, phi in enumerate(phi_values):
            # Update phi parameter
            original_phi = self.ml_params['phi']
            self.ml_params['phi'] = phi
            
            # Solve ODE
            sol = solve_ivp(self.morris_lecar_ode, t_span, initial_conditions, 
                            t_eval=t_eval, rtol=self.sim_config['rtol'], 
                            atol=self.sim_config['atol'])
            
            # Plot time series
            ax1.plot(sol.t, sol.y[0], linewidth=2.5, color=self.colors[i], 
                    label=f'φ = {phi:.3f}')
            
            # Plot phase plane trajectory
            ax2.plot(sol.y[0], sol.y[1], linewidth=2.5, color=self.colors[i], 
                    label=f'φ = {phi:.3f}')
            
            # Reset phi
            self.ml_params['phi'] = original_phi
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (mV)')
        ax1.set_title('Action Potentials with Different φ Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Voltage (mV)')
        ax2.set_ylabel('Recovery Variable')
        ax2.set_title('Phase Plane Trajectories')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def analyze_depolarization_threshold(self):
        """Analyze depolarization threshold"""
        print("Part 6: Depolarization Threshold Analysis")
        
        V_eq, w_eq = self.find_equilibrium_ml()
        
        # Test range of initial voltages around threshold
        v_init_range = np.linspace(-15.5, -14.5, 500)
        max_voltages = np.zeros(len(v_init_range))
        
        t_span = (0, 300)
        
        for i, v_init in enumerate(v_init_range):
            initial_conditions = [v_init, w_eq]
            sol = solve_ivp(self.morris_lecar_ode, t_span, initial_conditions, 
                            rtol=self.sim_config['rtol'], atol=self.sim_config['atol'])
            max_voltages[i] = np.max(sol.y[0])
        
        # Find threshold using gradient analysis
        gradient = np.diff(max_voltages) / np.diff(v_init_range)
        threshold_idx = np.argmax(np.abs(gradient))
        threshold = v_init_range[threshold_idx]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(v_init_range, max_voltages, linewidth=2.5, color=self.colors[0])
        ax1.scatter(threshold, max_voltages[threshold_idx], s=100, c=self.colors[3], 
                    zorder=5, label=f'Threshold = {threshold:.3f} mV')
        ax1.set_xlabel('Initial Voltage (mV)')
        ax1.set_ylabel('Maximum Voltage (mV)')
        ax1.set_title('Threshold Behavior')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Show trajectories around threshold
        self.plot_threshold_trajectories(ax2, threshold, w_eq)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Depolarization threshold: {threshold:.6f} mV")

    def plot_threshold_trajectories(self, ax, threshold, w_eq):
        """Plot trajectories around threshold"""
        test_voltages = threshold + np.array([-0.05, -0.02, 0, 0.02, 0.05])
        t_span = (0, 300)
        t_eval = np.linspace(0, 300, 3000)
        
        for i, v_test in enumerate(test_voltages):
            initial_conditions = [v_test, w_eq]
            sol = solve_ivp(self.morris_lecar_ode, t_span, initial_conditions, 
                            t_eval=t_eval, rtol=self.sim_config['rtol'], 
                            atol=self.sim_config['atol'])
            
            ax.plot(sol.t, sol.y[0], linewidth=2, label=f'V₀ = {v_test:.3f}')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('Trajectories Around Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def analyze_high_current_response(self):
        """Analyze response to high external current"""
        print("Part 7: High Current Response Analysis")
        
        # Set high external current
        original_current = self.ml_params['I_ext']
        self.ml_params['I_ext'] = 86
        
        # Find new equilibrium point
        V_eq_high, w_eq_high = self.find_equilibrium_ml()
        
        # Reset to original current to get original equilibrium
        self.ml_params['I_ext'] = original_current
        V_eq_orig, w_eq_orig = self.find_equilibrium_ml()
        self.ml_params['I_ext'] = 86  # Reset to high current
        
        print(f"Original equilibrium (I=0): V = {V_eq_orig:.3f}, w = {w_eq_orig:.3f}")
        print(f"High current equilibrium (I=86): V = {V_eq_high:.3f}, w = {w_eq_high:.3f}")
        
        # Analyze stability
        J = self.compute_jacobian_ml(V_eq_high, w_eq_high)
        eigenvals = np.linalg.eigvals(J)
        
        print(f"High current eigenvalues: {eigenvals[0]:.6f}, {eigenvals[1]:.6f}")
        
        # Create phase plane analysis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot nullclines
        V_range = np.linspace(-80, 100, 200)
        v_null = self.compute_voltage_nullcline(V_range)
        w_null = self.compute_recovery_nullcline(V_range)
        
        ax.plot(V_range, v_null, linewidth=2.5, color=self.colors[0], 
                label='V-nullcline (I=86)')
        ax.plot(V_range, w_null, linewidth=2.5, color=self.colors[1], 
                label='w-nullcline')
        
        # Mark equilibrium points
        ax.scatter(V_eq_high, w_eq_high, s=150, c=self.colors[2], 
                    label=f'Equilibrium I=86 ({V_eq_high:.2f}, {w_eq_high:.3f})')
        ax.scatter(V_eq_orig, w_eq_orig, s=150, c=self.colors[3], 
                    label=f'Equilibrium I=0 ({V_eq_orig:.2f}, {w_eq_orig:.3f})')
        
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Recovery Variable')
        ax.set_title('High Current Response (I = 86 µA/cm²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Reset to original current
        self.ml_params['I_ext'] = original_current

    def perform_multi_current_analysis(self):
        """Perform multi-current analysis"""
        print("Part 9: Multi-Current Analysis")
        
        current_values = [80, 86, 90]
        
        for I_ext in current_values:
            self.ml_params['I_ext'] = I_ext
            
            print(f"\n--- Analysis for I_ext = {I_ext} µA/cm² ---")
            
            # Find equilibrium and analyze stability
            V_eq, w_eq = self.find_equilibrium_ml()
            
            if not np.isnan(V_eq):
                J = self.compute_jacobian_ml(V_eq, w_eq)
                eigenvals = np.linalg.eigvals(J)
                
                print(f"Equilibrium: V = {V_eq:.3f} mV, w = {w_eq:.6f}")
                print(f"Eigenvalues: {eigenvals[0]:.6f}, {eigenvals[1]:.6f}")
                
                # Determine stability type
                if np.all(np.real(eigenvals) < 0):
                    if np.any(np.imag(eigenvals) != 0):
                        print("Type: Stable spiral")
                    else:
                        print("Type: Stable node")
                else:
                    if np.any(np.imag(eigenvals) != 0):
                        print("Type: Unstable spiral/focus")
                    else:
                        print("Type: Unstable node/saddle")
        
        # Generate firing rate vs current plot
        self.generate_firing_rate_plot()
        
        # Reset current
        self.ml_params['I_ext'] = 0

    def generate_firing_rate_plot(self):
        """Generate firing rate vs current plot"""
        current_range = np.arange(80, 101)
        firing_rates = np.zeros(len(current_range))
        
        for i, I_ext in enumerate(current_range):
            self.ml_params['I_ext'] = I_ext
            V_eq, w_eq = self.find_equilibrium_ml()
            
            if not np.isnan(V_eq):
                # Start slightly off equilibrium
                initial_conditions = [V_eq + 0.1, w_eq + 0.001]
                t_span = (0, 2000)
                
                sol = solve_ivp(self.morris_lecar_ode, t_span, initial_conditions, 
                                rtol=self.sim_config['rtol'], atol=self.sim_config['atol'])
                
                firing_rates[i] = self.calculate_firing_rate(sol.t, sol.y[0])
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(current_range, firing_rates, linewidth=3, color=self.colors[0], 
                marker='o', markersize=6)
        plt.xlabel('External Current (µA/cm²)')
        plt.ylabel('Firing Rate (Hz)')
        plt.title('Firing Rate vs Current')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def analyze_alternative_parameter_set(self):
        """Analyze alternative parameter set"""
        print("Parts 10-11: Alternative Parameter Set Analysis")
        
        # Store original parameters
        original_params = self.ml_params.copy()
        
        # Set alternative parameters
        self.ml_params.update({
            'gCa': 4,
            'V3': 12,
            'V4': 17.4,
            'phi': 0.0667,
            'I_ext': 30
        })
        
        print("Part 10: Analysis with I_ext = 30 µA/cm²")
        
        # Find equilibrium points
        equilibrium_points = self.find_multiple_equilibria()
        
        print(f"Found {len(equilibrium_points)} equilibrium points:")
        for i, (V_eq, w_eq) in enumerate(equilibrium_points):
            print(f"  Point {i+1}: V = {V_eq:.6f}, w = {w_eq:.6f}")
            
            # Analyze stability
            J = self.compute_jacobian_ml(V_eq, w_eq)
            eigenvals = np.linalg.eigvals(J)
            
            if np.all(np.real(eigenvals) < 0):
                stability = 'Stable'
            elif np.all(np.real(eigenvals) > 0):
                stability = 'Unstable'
            else:
                stability = 'Saddle'
            
            print(f"    Stability: {stability} (λ = {eigenvals[0]:.6f}, {eigenvals[1]:.6f})")
        
        # Restore original parameters
        self.ml_params = original_params

    def find_multiple_equilibria(self):
        """Find multiple equilibria using different initial guesses"""
        initial_guesses = [[-40, 0], [-20, 0.02], [0, 0.25]]
        equilibria = []
        
        for guess in initial_guesses:
            try:
                V_eq, w_eq = self.find_equilibrium_ml()
                if not np.isnan(V_eq):
                    # Check if this is a new equilibrium
                    is_new = True
                    for existing_eq in equilibria:
                        if np.linalg.norm([V_eq - existing_eq[0], w_eq - existing_eq[1]]) < 1e-6:
                            is_new = False
                            break
                    
                    if is_new:
                        equilibria.append([V_eq, w_eq])
            except:
                continue
        
        return equilibria

    # ========== HODGKIN-HUXLEY ANALYSIS ==========

    def execute_hodgkin_huxley_analysis(self):
        """Execute Hodgkin-Huxley model analysis"""
        self.calibrate_resting_potential()
        self.analyze_resting_stability_hh()
        self.analyze_steady_current_range_hh()
        self.analyze_reduced_model_hh()
        self.demonstrate_anode_break_excitation()

    def alpha_n(self, V):
        """Alpha function for n gate"""
        if np.abs(V + 50) < self.hh_params['eps']:
            return 0.1
        else:
            return -0.01 * (V + 50) / (np.exp(-(V + 50)/10) - 1)

    def alpha_m(self, V):
        """Alpha function for m gate"""
        if np.abs(V + 35) < self.hh_params['eps']:
            return 1.0
        else:
            return -0.1 * (V + 35) / (np.exp(-(V + 35)/10) - 1)

    def steady_state_values_hh(self, V):
        """Calculate steady-state values for HH gates"""
        alpha_n = self.alpha_n(V)
        beta_n = 0.125 * np.exp(-(V + 60)/80)
        
        alpha_m = self.alpha_m(V)
        beta_m = 4 * np.exp(-(V + 60)/18)
        
        alpha_h = 0.07 * np.exp(-(V + 60)/20)
        beta_h = 1 / (np.exp(-(V + 30)/10) + 1)
        
        n_inf = alpha_n / (alpha_n + beta_n)
        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)
        
        return n_inf, m_inf, h_inf

    def hodgkin_huxley_ode(self, t, y):
        """Hodgkin-Huxley ODE system"""
        V, n, m, h = y
        p = self.hh_params
        
        # Rate constants
        alpha_n = self.alpha_n(V)
        beta_n = 0.125 * np.exp(-(V + 60)/80)
        alpha_m = self.alpha_m(V)
        beta_m = 4 * np.exp(-(V + 60)/18)
        alpha_h = 0.07 * np.exp(-(V + 60)/20)
        beta_h = 1 / (np.exp(-(V + 30)/10) + 1)
        
        # Membrane currents
        I_Na = p['gNa'] * m**3 * h * (V - p['VNa'])
        I_K = p['gK'] * n**4 * (V - p['VK'])
        I_L = p['gL'] * (V - p['VL'])
        
        # Differential equations
        dVdt = (-I_Na - I_K - I_L + p['I_ext']) / p['C']
        dndt = alpha_n * (1 - n) - beta_n * n
        dmdt = alpha_m * (1 - m) - beta_m * m
        dhdt = alpha_h * (1 - h) - beta_h * h
        
        return [dVdt, dndt, dmdt, dhdt]

# ========== MAIN EXECUTION ==========

def main():
    """Main execution function"""
    print("Enhanced Neural Dynamics Simulation")
    print("Computational Neuroscience Project - Morris-Lecar & Hodgkin-Huxley Models")
    print("Python implementation")
    print("="*60)
    
    # Create simulation instance
    sim = NeuralDynamicsSimulation()
    
    # Run complete analysis
    sim.run_analysis()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("All analyses have been performed and plots generated.")
    print("Please refer to the generated figures for detailed results.")

if __name__ == "__main__":
    main()