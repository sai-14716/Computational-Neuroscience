import numpy as np
import matplotlib.pyplot as plt


"""
Computational Neuroscience Project IV (Python implementation)
============================================================

This script is a Python3 translation of the assignment described in:
file:///home/boss/Documents/Neuroscience/ASSIGNMENT-4/cn_project_iv_a25.pdf

It is organized so that a single run (main()) generates all required figures:
 - Inhomogeneous Poisson spike trains, PSTH, RMSE vs repetitions (Q1)
 - SP & L4 network with depressing synapses, no long-term plasticity (verNOp) and ODDBALL protocol (Q2–Q3)
 - Network with STDP at L4 synapses, long S/D sequence with 0.9/0.1 and ODDBALL probes (Q4)
 - Network with STDP, long S/D sequence with 0.5/0.5 (Q5)
"""


# ================================================================
# General utilities
# ================================================================


def ensure_rng(seed=0):
    return np.random.default_rng(seed)


# ================================================================
# Q1: Inhomogeneous Poisson spike trains
# ================================================================


def generate_random_lambda(T_ms=1000, mean_rate_hz=45.0, min_rate_hz=5.0, max_rate_hz=90.0, rng=None):
    """
    Generate a smooth random rate function lambda(t) in Hz over T_ms (1 ms resolution).
    Approximately around mean_rate_hz, always > 0.
    """
    if rng is None:
        rng = ensure_rng()

    t = np.arange(T_ms)
    # Start with white noise and low-pass filter to get smooth variations
    noise = rng.normal(0.0, 1.0, size=T_ms)
    lam = np.copy(noise)
    alpha = 0.01
    for i in range(1, T_ms):
        lam[i] = (1 - alpha) * lam[i - 1] + alpha * noise[i]
    lam = (lam - lam.mean()) / (lam.std() + 1e-9)
    lam = mean_rate_hz + 10.0 * lam  # add some modulation around mean
    lam = np.clip(lam, min_rate_hz, max_rate_hz)
    return lam  # Hz


def generate_inhom_poisson_spikes(lam_hz, n_reps, dt_ms=1.0, rng=None):
    """
    Generate inhomogeneous Poisson spike trains given lambda(t) in Hz.
    lam_hz: array of length T (Hz)
    n_reps: number of repetitions
    Returns boolean array spikes[n_reps, T]
    """
    if rng is None:
        rng = ensure_rng()

    T = lam_hz.shape[0]
    dt_s = dt_ms / 1000.0
    p = lam_hz * dt_s  # probability per bin
    p = np.clip(p, 0.0, 1.0)
    # Broadcast to reps
    rand = rng.random((n_reps, T))
    spikes = rand < p
    return spikes


def compute_psth(spikes, dt_ms=1.0):
    """
    spikes: bool/int array [n_reps, T]
    Returns psth[t] in spikes/s
    """
    n_reps, T = spikes.shape
    dt_s = dt_ms / 1000.0
    counts = spikes.sum(axis=0)
    rate = counts / (n_reps * dt_s)
    return rate


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def q1_inhom_poisson_demo(output_prefix="q1", rng=None):
    if rng is None:
        rng = ensure_rng()

    T_ms = 1000
    lam = generate_random_lambda(T_ms=T_ms, rng=rng)

    # Generate 320 repetitions
    n_total = 320
    spikes = generate_inhom_poisson_spikes(lam, n_total, rng=rng)

    # PSTH from first 100 repetitions
    spikes_100 = spikes[:100]
    psth_100 = compute_psth(spikes_100)

    # Plot lambda vs PSTH
    t = np.arange(T_ms)
    plt.figure(figsize=(8, 4))
    plt.plot(t, lam, label="lambda(t) [Hz]")
    plt.plot(t, psth_100, label="PSTH (100 reps)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Rate (spikes/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_lambda_vs_psth.png", dpi=200)
    plt.close()

    # RMSE vs number of repetitions
    rep_list = [10, 20, 40, 80, 160, 320]
    rmses = []
    for n in rep_list:
        psth_n = compute_psth(spikes[:n])
        rmses.append(rmse(psth_n, lam))

    plt.figure(figsize=(5, 4))
    plt.plot(rep_list, rmses, marker="o")
    plt.xlabel("Number of repetitions")
    plt.ylabel("RMSE (PSTH vs lambda)")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_rmse_vs_reps.png", dpi=200)
    plt.close()

    return lam, spikes


# ================================================================
# Synaptic depression (short-term plasticity)
# ================================================================


class DepressingSynapsePool:
    """
    Implements the recovered-effective-inactive model for a pool of identical
    synapses from a single presynaptic source.

    x_r + x_e + x_i = 1 (approximately).
    """

    def __init__(self, tau_re, tau_ei, tau_ir, dt_ms=1.0):
        self.dt = dt_ms
        self.tau_re = tau_re
        self.tau_ei = tau_ei
        self.tau_ir = tau_ir
        self.reset()

    def reset(self):
        self.x_r = 1.0
        self.x_e = 0.0
        self.x_i = 0.0

    def step(self, M):
        """
        One Euler step with presynaptic spike indicator M (0 or 1).
        """
        dt = self.dt
        dx_r = -M * self.x_r / self.tau_re + self.x_i / self.tau_ir
        dx_e = +M * self.x_r / self.tau_re - self.x_e / self.tau_ei
        dx_i = self.x_e / self.tau_ei - self.x_i / self.tau_ir
        self.x_r += dx_r * dt
        self.x_e += dx_e * dt
        self.x_i += dx_i * dt
        # Bound
        self.x_r = np.clip(self.x_r, 0.0, 1.0)
        self.x_e = np.clip(self.x_e, 0.0, 1.0)
        self.x_i = np.clip(self.x_i, 0.0, 1.0)
        # Optional renormalization
        s = self.x_r + self.x_e + self.x_i
        if s > 0:
            self.x_r /= s
            self.x_e /= s
            self.x_i /= s
        return self.x_e


# ================================================================
# LIF neuron with EPSP convolution and refractory drop
# ================================================================


class LIFNeuron:
    """
    Simple leaky integrate-and-fire neuron with:
    - Input as weighted sum of synaptic EPSP "g" terms (already convolved)
      times STP factors (x_e).
    - Threshold and spike generation.
    - Post-spike drop modeled via an exponential refractory kernel.
    - Additional 10% leak each time-step.
    """

    def __init__(self, threshold=0.05, beta=5.0, tau_ref=2.0, dt_ms=1.0):
        self.th = threshold
        self.beta = beta
        self.tau_ref = tau_ref
        self.dt = dt_ms
        self.reset()

    def reset(self):
        self.V = 0.0
        self.ref_kernel = 0.0
        self.spikes = []

    def step(self, t_idx, syn_input):
        """
        syn_input: scalar (sum of g*w*x_e for all inputs at this time)
        """
        dt = self.dt
        # Decay refractory kernel
        self.ref_kernel *= np.exp(-dt / self.tau_ref)

        # Base potential from current inputs (as per assignment equations)
        V_no_leak = syn_input

        # Subtract refractory effect and apply 10% leak
        V = (V_no_leak - self.ref_kernel) * 0.9

        spike = False
        if V >= self.th:
            spike = True
            self.spikes.append(t_idx)
            # start a new refractory drop
            self.ref_kernel += self.beta
            # Apply immediate drop at this step as well
            V = (V_no_leak - self.ref_kernel) * 0.9

        self.V = V
        return spike, V


# ================================================================
# STDP for L4 synapses
# ================================================================


class STDP:
    """
    Pair-based STDP using pre- and post-synaptic traces.
    d w / w = a_LTP * exp(-dt / tau_LTP) for dt>0 (pre then post)
    d w / w = -a_LTD * exp(dt / tau_LTD) for dt<0 (post then pre)

    Implementation via traces:
      - On pre spike at time t: w += w * (-a_LTD * post_trace)
      - On post spike at time t: w += w * ( a_LTP * pre_trace)

    where traces decay exponentially with respective time constants.
    """

    def __init__(self, a_ltp=0.015, tau_ltp=13.0, a_ltd=0.021, tau_ltd=20.0, dt_ms=1.0):
        self.a_ltp = a_ltp
        self.tau_ltp = tau_ltp
        self.a_ltd = a_ltd
        self.tau_ltd = tau_ltd
        self.dt = dt_ms
        self.reset()

    def reset(self):
        self.pre_trace = 0.0
        self.post_trace = 0.0

    def decay_traces(self):
        dt = self.dt
        self.pre_trace *= np.exp(-dt / self.tau_ltp)
        self.post_trace *= np.exp(-dt / self.tau_ltd)

    def on_pre(self, w):
        """
        Call when a presynaptic spike happens.
        Returns updated w.
        """
        self.decay_traces()
        dw_over_w = -self.a_ltd * self.post_trace
        w_new = w * (1.0 + dw_over_w)
        self.pre_trace += 1.0
        return w_new

    def on_post(self, w):
        """
        Call when a postsynaptic spike happens.
        Returns updated w.
        """
        self.decay_traces()
        dw_over_w = self.a_ltp * self.pre_trace
        w_new = w * (1.0 + dw_over_w)
        self.post_trace += 1.0
        return w_new


# ================================================================
# Thalamic input generation for S and D neurons
# ================================================================


def generate_S_D_spike_trains(total_T_ms, stim_onsets_ms, stim_ids, stim_duration_ms=50,
                              gap_ms=250, rate_S_S=10.0, rate_D_S=2.5,
                              rate_S_D=2.5, rate_D_D=10.0, rate_spont=0.5,
                              dt_ms=1.0, rng=None):
    """
    Generate Poisson spike trains for thalamic neurons S and D.

    total_T_ms: total time
    stim_onsets_ms: list/array of stimulus onset times in ms
    stim_ids: list/array of same length with 'S' or 'D'
    Each stimulus is stim_duration_ms long; between stimuli is spontaneous activity.
    """
    if rng is None:
        rng = ensure_rng()

    T = int(total_T_ms)
    dt_s = dt_ms / 1000.0
    rate_S = np.full(T, rate_spont)
    rate_D = np.full(T, rate_spont)

    for onset, sid in zip(stim_onsets_ms, stim_ids):
        start = int(onset)
        end = min(T, start + stim_duration_ms)
        if sid == 'S':
            rate_S[start:end] = rate_S_S
            rate_D[start:end] = rate_D_S
        else:
            rate_S[start:end] = rate_S_D
            rate_D[start:end] = rate_D_D

    p_S = np.clip(rate_S * dt_s, 0.0, 1.0)
    p_D = np.clip(rate_D * dt_s, 0.0, 1.0)

    S_spikes = rng.random(T) < p_S
    D_spikes = rng.random(T) < p_D
    return S_spikes, D_spikes


# ================================================================
# Network simulation (SP & L4, with/without plasticity)
# ================================================================


class SP_L4_Network:
    def __init__(self, dt_ms=1.0, tau_syn=10.0, tau_ref=2.0,
                 beta=5.0, threshold=0.05,
                 # Depression parameters
                 tau_re_th=0.9, tau_ei_th=10.0, tau_ir_th=5000.0,
                 tau_re_sp=0.9, tau_ei_sp=27.0, tau_ir_sp=5000.0,
                 # Initial weights
                 w_S_SP_init=0.2, w_D_SP_init=0.2,
                 w_S_L4_init=0.02, w_D_L4_init=0.02,
                 w_SP_L4_init=0.11,
                 # Weight bounds
                 w_th_L4_min=0.0001, w_th_L4_max=0.4,
                 w_SP_L4_min=0.0001, w_SP_L4_max=0.11,
                 with_plasticity=True):
        self.dt = dt_ms
        self.tau_syn = tau_syn

        # Neurons
        self.SP = LIFNeuron(threshold=threshold, beta=beta, tau_ref=tau_ref, dt_ms=dt_ms)
        self.L4 = LIFNeuron(threshold=threshold, beta=beta, tau_ref=tau_ref, dt_ms=dt_ms)

        # Depression pools (shared for all synapses from same pre-source)
        self.depr_S = DepressingSynapsePool(tau_re_th, tau_ei_th, tau_ir_th, dt_ms)
        self.depr_D = DepressingSynapsePool(tau_re_th, tau_ei_th, tau_ir_th, dt_ms)
        self.depr_SP = DepressingSynapsePool(tau_re_sp, tau_ei_sp, tau_ir_sp, dt_ms)

        # Weights
        self.w_S_SP = w_S_SP_init
        self.w_D_SP = w_D_SP_init
        self.w_S_L4 = w_S_L4_init
        self.w_D_L4 = w_D_L4_init
        self.w_SP_L4 = w_SP_L4_init

        self.w_th_L4_min = w_th_L4_min
        self.w_th_L4_max = w_th_L4_max
        self.w_SP_L4_min = w_SP_L4_min
        self.w_SP_L4_max = w_SP_L4_max

        self.with_plasticity = with_plasticity

        # STDP for L4 synapses
        self.stdp_S_L4 = STDP(dt_ms=dt_ms)
        self.stdp_D_L4 = STDP(dt_ms=dt_ms)
        self.stdp_SP_L4 = STDP(dt_ms=dt_ms)

        # EPSP kernels g (for S, D, SP presyn)
        self.g_S = 0.0
        self.g_D = 0.0
        self.g_SP = 0.0

    def reset_state(self):
        self.SP.reset()
        self.L4.reset()
        self.depr_S.reset()
        self.depr_D.reset()
        self.depr_SP.reset()
        self.g_S = 0.0
        self.g_D = 0.0
        self.g_SP = 0.0
        self.stdp_S_L4.reset()
        self.stdp_D_L4.reset()
        self.stdp_SP_L4.reset()

    def step(self, t_idx, spike_S, spike_D):
        """
        One simulation step.
        spike_S, spike_D: booleans for thalamic spikes at this time index.
        Returns:
          spike_SP, spike_L4, V_SP, V_L4, (w_S_L4, w_D_L4, w_SP_L4)
        """
        dt = self.dt

        # Update EPSP kernels with 1 ms synaptic delay:
        # use spikes from previous time step in g(t)
        self.g_S *= np.exp(-dt / self.tau_syn)
        self.g_D *= np.exp(-dt / self.tau_syn)
        self.g_SP *= np.exp(-dt / self.tau_syn)

        if hasattr(self, "_prev_spike_S") and self._prev_spike_S:
            self.g_S += 1.0
        if hasattr(self, "_prev_spike_D") and self._prev_spike_D:
            self.g_D += 1.0
        if hasattr(self, "_prev_spike_SP") and self._prev_spike_SP:
            self.g_SP += 1.0

        # Depression dynamics
        x_e_S = self.depr_S.step(1.0 if spike_S else 0.0)
        x_e_D = self.depr_D.step(1.0 if spike_D else 0.0)
        # SP presyn will be updated after we know spike_SP

        # SP neuron: inputs from S and D thalamic synapses
        syn_SP = self.g_S * self.w_S_SP * x_e_S + self.g_D * self.w_D_SP * x_e_D
        spike_SP, V_SP = self.SP.step(t_idx, syn_SP)

        # Now update SP depression pool with SP spike (for its synapse on L4)
        x_e_SP = self.depr_SP.step(1.0 if spike_SP else 0.0)

        # L4 neuron: inputs from S, D, SP
        syn_L4 = (
            self.g_S * self.w_S_L4 * x_e_S
            + self.g_D * self.w_D_L4 * x_e_D
            + self.g_SP * self.w_SP_L4 * x_e_SP
        )
        spike_L4, V_L4 = self.L4.step(t_idx, syn_L4)

        # STDP updates if enabled
        if self.with_plasticity:
            # Pre events
            if spike_S:
                self.w_S_L4 = self.stdp_S_L4.on_pre(self.w_S_L4)
            if spike_D:
                self.w_D_L4 = self.stdp_D_L4.on_pre(self.w_D_L4)
            if spike_SP:
                self.w_SP_L4 = self.stdp_SP_L4.on_pre(self.w_SP_L4)

            # Post event
            if spike_L4:
                self.w_S_L4 = self.stdp_S_L4.on_post(self.w_S_L4)
                self.w_D_L4 = self.stdp_D_L4.on_post(self.w_D_L4)
                self.w_SP_L4 = self.stdp_SP_L4.on_post(self.w_SP_L4)

            # Enforce weight bounds
            self.w_S_L4 = np.clip(self.w_S_L4, self.w_th_L4_min, self.w_th_L4_max)
            self.w_D_L4 = np.clip(self.w_D_L4, self.w_th_L4_min, self.w_th_L4_max)
            self.w_SP_L4 = np.clip(self.w_SP_L4, self.w_SP_L4_min, self.w_SP_L4_max)

        # Save this step's spikes for next-step delayed EPSPs
        self._prev_spike_S = bool(spike_S)
        self._prev_spike_D = bool(spike_D)
        self._prev_spike_SP = bool(spike_SP)

        return spike_SP, spike_L4, V_SP, V_L4, (self.w_S_L4, self.w_D_L4, self.w_SP_L4)


def run_oddball_verNOp(n_reps=50, tau_re_th=0.9, tau_re_sp=0.9, rng=None,
                       output_prefix="q2_q3"):
    """
    Run the ODDBALL protocol (15 stimuli, 8th is D; others S) using
    a network version without long-term plasticity (verNOp).
    Returns spike trains and PSTHs for SP and L4.
    """
    if rng is None:
        rng = ensure_rng()

    dt_ms = 1.0
    stim_dur = 50
    gap = 250
    n_stim = 15
    total_T_ms = n_stim * (stim_dur + gap)

    # Build ODDBALL stimulus sequence
    stim_onsets = np.arange(0, n_stim * (stim_dur + gap), stim_dur + gap)
    stim_ids = np.array(['S'] * n_stim, dtype=object)
    stim_ids[7] = 'D'  # 8th stimulus is deviant

    # Collect all spikes across reps for PSTH
    T = int(total_T_ms)
    all_spikes_SP = np.zeros((n_reps, T), dtype=bool)
    all_spikes_L4 = np.zeros((n_reps, T), dtype=bool)

    for rep in range(n_reps):
        network = SP_L4_Network(
            dt_ms=dt_ms,
            tau_re_th=tau_re_th,
            tau_ei_th=10.0,
            tau_ir_th=5000.0,
            tau_re_sp=tau_re_sp,
            tau_ei_sp=27.0,
            tau_ir_sp=5000.0,
            with_plasticity=False,
        )
        network.reset_state()

        S_spikes, D_spikes = generate_S_D_spike_trains(
            total_T_ms=total_T_ms,
            stim_onsets_ms=stim_onsets,
            stim_ids=stim_ids,
            stim_duration_ms=stim_dur,
            gap_ms=gap,
            rng=rng,
        )

        for t in range(T):
            spike_SP, spike_L4, V_SP, V_L4, _ = network.step(t, S_spikes[t], D_spikes[t])
            all_spikes_SP[rep, t] = spike_SP
            all_spikes_L4[rep, t] = spike_L4

    # PSTHs with 10 ms bin size
    bin_ms = 10
    n_bins = T // bin_ms

    def binned_psth(spikes):
        n_reps, T = spikes.shape
        spikes = spikes[:, : n_bins * bin_ms]
        spikes_reshaped = spikes.reshape(n_reps, n_bins, bin_ms)
        counts = spikes_reshaped.sum(axis=2)  # per bin
        dt_s = bin_ms / 1000.0
        rate = counts.mean(axis=0) / dt_s
        return rate

    psth_SP = binned_psth(all_spikes_SP)
    psth_L4 = binned_psth(all_spikes_L4)
    t_bins = np.arange(n_bins) * bin_ms

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.bar(t_bins, psth_SP, width=bin_ms, align="edge")
    plt.ylabel("SP rate (sp/s)")
    plt.subplot(2, 1, 2)
    plt.bar(t_bins, psth_L4, width=bin_ms, align="edge")
    plt.ylabel("L4 rate (sp/s)")
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_psth_tau_re_{tau_re_th}_{tau_re_sp}.png", dpi=200)
    plt.close()

    return {
        "all_spikes_SP": all_spikes_SP,
        "all_spikes_L4": all_spikes_L4,
        "psth_SP": psth_SP,
        "psth_L4": psth_L4,
        "t_bins": t_bins,
    }


def q3_tau_re_variations(rng=None):
    """
    Run ODDBALL protocol for different tau_re values:
      a) 1000 ms ; b) 3000 ms ; c) 10000 ms
    for both thalamic and SP synapses.
    """
    if rng is None:
        rng = ensure_rng()

    tau_res = [1000.0, 3000.0, 10000.0]
    results = {}
    for tau in tau_res:
        res = run_oddball_verNOp(
            n_reps=50,
            tau_re_th=tau,
            tau_re_sp=tau,
            rng=rng,
            output_prefix=f"q3_tau_{int(tau)}",
        )
        results[tau] = res
    return results


# ================================================================
# Long plasticity runs (Q4, Q5)
# ================================================================


def build_random_stim_sequence_for_long_run(total_time_s=60 * 60,
                                            stim_duration_ms=50,
                                            gap_ms=250,
                                            p_S=0.9,
                                            p_D=0.1,
                                            rng=None):
    """
    Build random S/D stimulus sequence over total_time_s seconds.
    Stim start-to-start interval = stim_duration_ms + gap_ms (300 ms).
    """
    if rng is None:
        rng = ensure_rng()

    dt_ms = 1.0
    T_ms = int(total_time_s * 1000)
    isi = stim_duration_ms + gap_ms
    n_stim = T_ms // isi
    stim_onsets = np.arange(0, n_stim * isi, isi)
    # Random S/D with given probabilities
    stim_ids = rng.choice(['S', 'D'], size=n_stim, p=[p_S, p_D])
    return T_ms, stim_onsets, stim_ids


def run_long_plasticity_run(total_time_s=60 * 60,
                            p_S=0.9,
                            p_D=0.1,
                            chunk_n_stim=100,
                            rng=None,
                            output_prefix="q4"):
    """
    Run long plasticity simulation (with STDP enabled) for given S/D probabilities.
    Returns trajectories of the three L4 synaptic weights over time.
    """
    if rng is None:
        rng = ensure_rng()

    stim_duration_ms = 50
    gap_ms = 250
    isi = stim_duration_ms + gap_ms
    T_ms, stim_onsets_all, stim_ids_all = build_random_stim_sequence_for_long_run(
        total_time_s=total_time_s,
        stim_duration_ms=stim_duration_ms,
        gap_ms=gap_ms,
        p_S=p_S,
        p_D=p_D,
        rng=rng,
    )

    network = SP_L4_Network(with_plasticity=True)
    network.reset_state()

    n_stim = len(stim_onsets_all)
    T_total = int(T_ms)
    w_S_L4_traj = []
    w_D_L4_traj = []
    w_SP_L4_traj = []
    t_traj = []

    # Pre-build rate templates so we can generate spikes chunk by chunk
    for chunk_start_idx in range(0, n_stim, chunk_n_stim):
        chunk_end_idx = min(n_stim, chunk_start_idx + chunk_n_stim)
        stim_onsets = stim_onsets_all[chunk_start_idx:chunk_end_idx]
        stim_ids = stim_ids_all[chunk_start_idx:chunk_end_idx]
        if len(stim_onsets) == 0:
            break
        chunk_T_ms = stim_onsets[-1] + isi - stim_onsets[0]
        T_chunk = int(chunk_T_ms)

        S_spikes, D_spikes = generate_S_D_spike_trains(
            total_T_ms=T_chunk,
            stim_onsets_ms=stim_onsets - stim_onsets[0],
            stim_ids=stim_ids,
            stim_duration_ms=stim_duration_ms,
            gap_ms=gap_ms,
            rng=rng,
        )

        t0 = stim_onsets[0]
        for k in range(T_chunk):
            t_glob = t0 + k
            if t_glob >= T_total:
                break
            spike_SP, spike_L4, V_SP, V_L4, (wS, wD, wSP) = network.step(
                t_glob, S_spikes[k], D_spikes[k]
            )
            if t_glob % isi == 0:
                # sample weights at each stimulus onset
                t_traj.append(t_glob)
                w_S_L4_traj.append(wS)
                w_D_L4_traj.append(wD)
                w_SP_L4_traj.append(wSP)

    t_traj = np.array(t_traj)
    w_S_L4_traj = np.array(w_S_L4_traj)
    w_D_L4_traj = np.array(w_D_L4_traj)
    w_SP_L4_traj = np.array(w_SP_L4_traj)

    plt.figure(figsize=(7, 4))
    plt.plot(t_traj / 1000.0 / 60.0, w_S_L4_traj, label="w_S->L4")
    plt.plot(t_traj / 1000.0 / 60.0, w_D_L4_traj, label="w_D->L4")
    plt.plot(t_traj / 1000.0 / 60.0, w_SP_L4_traj, label="w_SP->L4")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Synaptic weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_weights.png", dpi=200)
    plt.close()

    return {
        "t_traj": t_traj,
        "w_S_L4_traj": w_S_L4_traj,
        "w_D_L4_traj": w_D_L4_traj,
        "w_SP_L4_traj": w_SP_L4_traj,
        "network": network,
    }


def probe_oddball_at_times(weight_times, network_template_params=None,
                           probe_times_ms=None, rng=None,
                           output_prefix="q4_probe"):
    """
    Given a trajectory of weight values over time (weight_times dict from run_long_plasticity_run),
    pick several time points and run ODDBALL protocol (verNOp, i.e., no further plasticity)
    with the corresponding weights as initial values.
    """
    if rng is None:
        rng = ensure_rng()

    if network_template_params is None:
        network_template_params = {}

    t_traj = weight_times["t_traj"]
    wS_traj = weight_times["w_S_L4_traj"]
    wD_traj = weight_times["w_D_L4_traj"]
    wSP_traj = weight_times["w_SP_L4_traj"]

    if probe_times_ms is None:
        # pick ~5 evenly spaced points
        idxs = np.linspace(0, len(t_traj) - 1, 5, dtype=int)
    else:
        # find closest indices
        idxs = [np.argmin(np.abs(t_traj - pt)) for pt in probe_times_ms]

    results = {}
    for i, idx in enumerate(idxs):
        wS = wS_traj[idx]
        wD = wD_traj[idx]
        wSP = wSP_traj[idx]

        # Run ODDBALL with these fixed weights and no plasticity
        res = run_oddball_with_custom_weights(
            w_S_L4=wS,
            w_D_L4=wD,
            w_SP_L4=wSP,
            rng=rng,
            output_prefix=f"{output_prefix}_probe_{i}",
            **network_template_params,
        )
        results[t_traj[idx]] = res

    return results


def run_oddball_with_custom_weights(w_S_L4, w_D_L4, w_SP_L4,
                                    n_reps=50,
                                    tau_re_th=0.9,
                                    tau_re_sp=0.9,
                                    rng=None,
                                    output_prefix="q4_oddball_custom"):
    """
    Same as run_oddball_verNOp but with custom initial L4 weights.
    """
    if rng is None:
        rng = ensure_rng()

    dt_ms = 1.0
    stim_dur = 50
    gap = 250
    n_stim = 15
    total_T_ms = n_stim * (stim_dur + gap)

    stim_onsets = np.arange(0, n_stim * (stim_dur + gap), stim_dur + gap)
    stim_ids = np.array(['S'] * n_stim, dtype=object)
    stim_ids[7] = 'D'

    T = int(total_T_ms)
    all_spikes_SP = np.zeros((n_reps, T), dtype=bool)
    all_spikes_L4 = np.zeros((n_reps, T), dtype=bool)

    for rep in range(n_reps):
        network = SP_L4_Network(
            dt_ms=dt_ms,
            tau_re_th=tau_re_th,
            tau_ei_th=10.0,
            tau_ir_th=5000.0,
            tau_re_sp=tau_re_sp,
            tau_ei_sp=27.0,
            tau_ir_sp=5000.0,
            with_plasticity=False,
            w_S_L4_init=w_S_L4,
            w_D_L4_init=w_D_L4,
            w_SP_L4_init=w_SP_L4,
        )
        network.reset_state()

        S_spikes, D_spikes = generate_S_D_spike_trains(
            total_T_ms=total_T_ms,
            stim_onsets_ms=stim_onsets,
            stim_ids=stim_ids,
            stim_duration_ms=stim_dur,
            gap_ms=gap,
            rng=rng,
        )

        for t in range(T):
            spike_SP, spike_L4, V_SP, V_L4, _ = network.step(t, S_spikes[t], D_spikes[t])
            all_spikes_SP[rep, t] = spike_SP
            all_spikes_L4[rep, t] = spike_L4

    # PSTHs 10 ms
    bin_ms = 10
    n_bins = T // bin_ms

    def binned_psth(spikes):
        n_reps, T = spikes.shape
        spikes = spikes[:, : n_bins * bin_ms]
        spikes_reshaped = spikes.reshape(n_reps, n_bins, bin_ms)
        counts = spikes_reshaped.sum(axis=2)
        dt_s = bin_ms / 1000.0
        rate = counts.mean(axis=0) / dt_s
        return rate

    psth_SP = binned_psth(all_spikes_SP)
    psth_L4 = binned_psth(all_spikes_L4)
    t_bins = np.arange(n_bins) * bin_ms

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.bar(t_bins, psth_SP, width=bin_ms, align="edge")
    plt.ylabel("SP rate (sp/s)")
    plt.subplot(2, 1, 2)
    plt.bar(t_bins, psth_L4, width=bin_ms, align="edge")
    plt.ylabel("L4 rate (sp/s)")
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_psth.png", dpi=200)
    plt.close()

    return {
        "all_spikes_SP": all_spikes_SP,
        "all_spikes_L4": all_spikes_L4,
        "psth_SP": psth_SP,
        "psth_L4": psth_L4,
        "t_bins": t_bins,
        "w_S_L4": w_S_L4,
        "w_D_L4": w_D_L4,
        "w_SP_L4": w_SP_L4,
    }


# ================================================================
# Main driver to generate all figures
# ================================================================


def main():
    rng = ensure_rng(42)

    # Q1: inhomogeneous Poisson and RMSE
    q1_inhom_poisson_demo(output_prefix="q1", rng=rng)

    # Q2: ODDBALL with initial parameters (depressing synapses, no plasticity)
    run_oddball_verNOp(n_reps=50, tau_re_th=0.9, tau_re_sp=0.9, rng=rng, output_prefix="q2")

    # Q3: Different tau_re values
    q3_tau_re_variations(rng=rng)

    # Q4: Long run with p_S=0.9, p_D=0.1
    long_res_0_9 = run_long_plasticity_run(
        total_time_s=60 * 60,
        p_S=0.9,
        p_D=0.1,
        chunk_n_stim=100,
        rng=rng,
        output_prefix="q4_0p9",
    )

    probe_oddball_at_times(
        weight_times=long_res_0_9,
        probe_times_ms=None,
        rng=rng,
        output_prefix="q4_0p9",
    )

    # Q5: Long run with p_S=0.5, p_D=0.5
    run_long_plasticity_run(
        total_time_s=60 * 60,
        p_S=0.5,
        p_D=0.5,
        chunk_n_stim=100,
        rng=rng,
        output_prefix="q5_0p5",
    )


if __name__ == "__main__":
    main()


