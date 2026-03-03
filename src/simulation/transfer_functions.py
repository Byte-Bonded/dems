"""
Transfer Function Models for DEMS Control Loops

Extracts s-domain transfer functions from the time-domain dynamic models in
dynamics.py.  Uses scipy.signal for representation and Bode/Nyquist analysis.

Models:
- AVR:  IEEE Type 1 Excitation System (IEEET1) — IEEE Std 421.5-2016 §5.1
- Governor:  IEEE TGOV1 — NERC PPMV compliant
- PSS:  PSS1A — IEEE Std 421.5-2016 §8.1
- AGC:  PI controller with deadband
- SMIB:  Single-Machine Infinite-Bus composite model

IEEE 421.5-2016 requirements:
    Gain Margin  ≥ 6 dB
    Phase Margin ≥ 30°
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class StabilityMargins:
    """Gain and phase margin results per IEEE 421.5-2016."""
    gain_margin_db: float
    phase_margin_deg: float
    gain_crossover_freq_rad: float  # rad/s where |G|=1
    phase_crossover_freq_rad: float  # rad/s where angle(G)=-180°
    ieee_compliant: bool  # GM≥6dB AND PM≥30°


def compute_margins(sys: signal.TransferFunction) -> StabilityMargins:
    """Compute gain and phase margins for a SISO transfer function.

    Uses scipy.signal to evaluate frequency response, then finds
    critical crossover frequencies where |G(jω)|=1 and ∠G(jω)=-180°.

    Returns:
        StabilityMargins with GM (dB), PM (deg), crossover frequencies.
    """
    w = np.logspace(-3, 4, 10000)
    w, H = signal.freqresp(sys, w)
    mag = np.abs(H)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))

    # Phase crossover: where phase crosses -180° → gain margin
    gm_db = np.inf
    phase_cross_freq = 0.0
    for i in range(len(phase_deg) - 1):
        if phase_deg[i] > -180.0 and phase_deg[i + 1] <= -180.0:
            # Linear interpolation
            frac = (-180.0 - phase_deg[i]) / (phase_deg[i + 1] - phase_deg[i])
            w_cross = w[i] + frac * (w[i + 1] - w[i])
            mag_cross = mag[i] + frac * (mag[i + 1] - mag[i])
            if mag_cross > 0:
                gm_db = -20.0 * np.log10(mag_cross)
                phase_cross_freq = w_cross
            break

    # Gain crossover: where magnitude crosses 1.0 (0 dB) → phase margin
    pm_deg = -180.0
    gain_cross_freq = 0.0
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-20))
    for i in range(len(mag_db) - 1):
        if mag_db[i] > 0 and mag_db[i + 1] <= 0:
            frac = (0.0 - mag_db[i]) / (mag_db[i + 1] - mag_db[i])
            w_cross = w[i] + frac * (w[i + 1] - w[i])
            phase_cross = phase_deg[i] + frac * (phase_deg[i + 1] - phase_deg[i])
            pm_deg = 180.0 + phase_cross
            gain_cross_freq = w_cross
            break

    return StabilityMargins(
        gain_margin_db=gm_db,
        phase_margin_deg=pm_deg,
        gain_crossover_freq_rad=gain_cross_freq,
        phase_crossover_freq_rad=phase_cross_freq,
        ieee_compliant=(gm_db >= 6.0 and pm_deg >= 30.0),
    )


# ======================== INDIVIDUAL TRANSFER FUNCTIONS ======================== #

def avr_transfer_function(
    KA: float = 200.0,
    TA: float = 0.02,
    KE: float = 1.0,
    TE: float = 0.5,
    KF: float = 0.03,
    TF: float = 1.0,
) -> Dict[str, signal.TransferFunction]:
    """IEEE Type 1 Exciter (IEEET1) transfer functions.

    Block diagram (IEEE Std 421.5-2016, Fig 5-1):

        ΔVref → [+] → [KA/(1+sTA)] → [1/(KE+sTE)] → Efd
                [-]                                     |
                 ↑←──── [sKF/(1+sTF)] ←────────────────┘

    Open-loop (without feedback):
        G_ol(s) = KA / ((1+sTA)(KE+sTE))

    Feedback path:
        H(s) = sKF / (1+sTF)

    Closed-loop:
        G_cl(s) = G_ol / (1 + G_ol × H)

    Returns:
        Dict with 'open_loop', 'feedback', 'closed_loop' TransferFunctions.
    """
    # Forward path: KA/((1+sTA)(KE+sTE))
    # (1+sTA) = [TA, 1]  ;  (KE+sTE) = [TE, KE]
    num_fwd = [KA]
    den_fwd = np.polymul([TA, 1.0], [TE, KE])

    # Feedback: sKF/(1+sTF)
    num_fb = [KF, 0.0]
    den_fb = [TF, 1.0]

    G_ol = signal.TransferFunction(num_fwd, den_fwd)

    # Loop transfer function: G_ol × H
    num_loop = np.polymul(num_fwd, num_fb)
    den_loop = np.polymul(den_fwd, den_fb)
    G_loop = signal.TransferFunction(num_loop, den_loop)

    # Closed-loop: G_ol / (1 + G_ol × H)
    # = num_fwd*den_fb / (den_fwd*den_fb + num_fwd*num_fb)
    num_cl = np.polymul(num_fwd, den_fb)
    den_cl_1 = np.polymul(den_fwd, den_fb)
    den_cl_2 = np.polymul(num_fwd, num_fb)
    # Pad to same length for addition
    max_len = max(len(den_cl_1), len(den_cl_2))
    den_cl_1_pad = np.pad(den_cl_1, (max_len - len(den_cl_1), 0))
    den_cl_2_pad = np.pad(den_cl_2, (max_len - len(den_cl_2), 0))
    den_cl = den_cl_1_pad + den_cl_2_pad

    G_cl = signal.TransferFunction(num_cl, den_cl)

    return {
        'open_loop': G_ol,
        'feedback': signal.TransferFunction(num_fb, den_fb),
        'loop': G_loop,
        'closed_loop': G_cl,
    }


def governor_transfer_function(
    R: float = 0.05,
    TG: float = 0.2,
    TT: float = 0.5,
    Dt: float = 0.05,
) -> Dict[str, signal.TransferFunction]:
    """IEEE TGOV1 Governor-Turbine transfer function.

    Block diagram:
        Δω → [-1/R] → [1/(1+sTG)] → valve → [1/(1+sTT)] → ΔPm
                                                              |
                                            [-Dt] ← ─────────┘

    Open-loop (Δω → ΔPm, without Dt feedback):
        G_gov(s) = (1/R) / ((1+sTG)(1+sTT))

    With turbine damping:
        G_gov_d(s) = G_gov(s) - Dt

    Returns:
        Dict with 'open_loop', 'with_damping' TransferFunctions.
    """
    # G(s) = (1/R) / ((1+sTG)(1+sTT))
    num = [1.0 / R]
    den = np.polymul([TG, 1.0], [TT, 1.0])

    G_ol = signal.TransferFunction(num, den)

    # With damping: (1/R - Dt*(1+sTG)*(1+sTT)) / ((1+sTG)*(1+sTT))
    num_d = np.array(num)
    num_d_padded = np.pad(num_d, (len(den) - len(num_d), 0))
    num_with_damping = num_d_padded - Dt * np.array(den)

    G_damped = signal.TransferFunction(num_with_damping, den)

    return {
        'open_loop': G_ol,
        'with_damping': G_damped,
    }


def pss_transfer_function(
    K_pss: float = 5.0,
    T_washout: float = 1.41,
    T_lead1: float = 0.154,
    T_lag1: float = 0.033,
    T_lead2: float = 0.154,
    T_lag2: float = 0.033,
) -> signal.TransferFunction:
    """PSS1A transfer function (IEEE Std 421.5-2016, §8.1).

    G_PSS(s) = K_pss × [sT_w/(1+sT_w)] × [(1+sT_lead1)/(1+sT_lag1)]
                                           × [(1+sT_lead2)/(1+sT_lag2)]

    - Washout: high-pass filter, blocks steady-state offset
    - Lead-lag: provides phase lead at inter-area oscillation frequencies

    Returns:
        Single TransferFunction object.
    """
    # Washout: sT/(1+sT) → num=[T,0], den=[T,1]
    num_w = [T_washout, 0.0]
    den_w = [T_washout, 1.0]

    # Lead-lag 1: (1+sT_lead)/(1+sT_lag) → num=[T_lead,1], den=[T_lag,1]
    num_ll1 = [T_lead1, 1.0]
    den_ll1 = [T_lag1, 1.0]

    # Lead-lag 2
    num_ll2 = [T_lead2, 1.0]
    den_ll2 = [T_lag2, 1.0]

    # Overall: K × washout × LL1 × LL2
    num = K_pss * np.polymul(np.polymul(num_w, num_ll1), num_ll2)
    den = np.polymul(np.polymul(den_w, den_ll1), den_ll2)

    return signal.TransferFunction(num, den)


def agc_transfer_function(
    K_agc: float = 0.5,
    T_agc: float = 4.0,
    beta: float = 1000.0,
) -> signal.TransferFunction:
    """AGC / Load-Frequency Control (LFC) PI transfer function.

    G_AGC(s) = -(K_agc + 1/(s*T_agc)) × β

    Proportional-Integral:  K_p = K_agc * β,  K_i = β / T_agc

    PI transfer: G(s) = -(K_p*s + K_i) / s = -(K_agc*beta*s + beta/T_agc) / s

    Returns:
        TransferFunction for ACE → ΔPref adjustment.
    """
    Kp = K_agc * beta
    Ki = beta / T_agc
    # G(s) = -(Kp*s + Ki)/s
    num = [-Kp, -Ki]
    den = [1.0, 0.0]

    return signal.TransferFunction(num, den)


def swing_equation_tf(
    H: float = 3.5,
    D: float = 2.0,
    f0: float = 50.0,
) -> signal.TransferFunction:
    """Swing equation transfer function (linearised).

    Δω(s) / ΔP(s) = 1 / (2Hs + D)

    where ΔP = Pm - Pe (pu), Δω (pu).
    Frequency: Δf = f0 × Δω.

    Returns:
        TransferFunction from ΔP(pu) to Δω(pu).
    """
    num = [1.0]
    den = [2.0 * H, D]
    return signal.TransferFunction(num, den)


def smib_composite_model(
    H: float = 3.5,
    D: float = 2.0,
    f0: float = 50.0,
    R: float = 0.05,
    TG: float = 0.2,
    TT: float = 0.5,
    Dt: float = 0.05,
    KA: float = 200.0,
    TA: float = 0.02,
    KE: float = 1.0,
    TE: float = 0.5,
    KF: float = 0.03,
    TF: float = 1.0,
    K_pss: float = 5.0,
    T_washout: float = 1.41,
    T_lead1: float = 0.154,
    T_lag1: float = 0.033,
    T_lead2: float = 0.154,
    T_lag2: float = 0.033,
) -> Dict[str, object]:
    """Build a complete Single-Machine Infinite-Bus (SMIB) composite model.

    Combines swing equation, governor, AVR, and PSS into a closed-loop system:

        ΔPL → [Swing] → Δω → [Governor] → ΔPm → [Swing] (feedback)
                          └──→ [PSS] → Vpss → [AVR] → ΔEfd → ΔPe (via network)

    Returns dict with individual and composite transfer functions plus margins.
    """
    G_swing = swing_equation_tf(H, D, f0)
    G_gov = governor_transfer_function(R, TG, TT, Dt)
    G_avr = avr_transfer_function(KA, TA, KE, TE, KF, TF)
    G_pss = pss_transfer_function(K_pss, T_washout, T_lead1, T_lag1, T_lead2, T_lag2)

    # Compute margins for key loops
    avr_margins = compute_margins(G_avr['loop'])
    gov_margins = compute_margins(G_gov['open_loop'])

    # Governor-swing closed loop:
    # Δω = 1/(2Hs+D) × (ΔPm_ext - ΔPe + ΔPm_gov)
    # where ΔPm_gov = -Δω × (1/R) / ((1+sTG)(1+sTT))
    # Open-loop transfer: G_swing × G_gov_ol = 1/((2Hs+D)×R×(1+sTG)(1+sTT))
    num_gov_swing = G_swing.num * G_gov['open_loop'].num[0] if len(G_gov['open_loop'].num) == 1 else np.polymul(G_swing.num, G_gov['open_loop'].num)
    den_gov_swing = np.polymul(G_swing.den, G_gov['open_loop'].den)
    G_gov_swing_loop = signal.TransferFunction(num_gov_swing, den_gov_swing)
    gov_swing_margins = compute_margins(G_gov_swing_loop)

    return {
        'swing': G_swing,
        'governor': G_gov,
        'avr': G_avr,
        'pss': G_pss,
        'gov_swing_loop': G_gov_swing_loop,
        'margins': {
            'avr': avr_margins,
            'governor': gov_margins,
            'gov_swing': gov_swing_margins,
        },
    }


# ======================== BODE / NYQUIST DATA GENERATION ======================== #

def bode_data(
    sys: signal.TransferFunction,
    w_range: Tuple[float, float] = (1e-3, 1e4),
    n_points: int = 2000,
) -> Dict[str, np.ndarray]:
    """Compute Bode plot data for a transfer function.

    Returns:
        Dict with 'omega' (rad/s), 'magnitude_db', 'phase_deg'.
    """
    w = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), n_points)
    w_out, H = signal.freqresp(sys, w)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-20))
    phase_deg = np.degrees(np.unwrap(np.angle(H)))

    return {
        'omega': w_out,
        'magnitude_db': mag_db,
        'phase_deg': phase_deg,
    }


def nyquist_data(
    sys: signal.TransferFunction,
    w_range: Tuple[float, float] = (1e-3, 1e4),
    n_points: int = 2000,
) -> Dict[str, np.ndarray]:
    """Compute Nyquist plot data for a transfer function.

    Returns:
        Dict with 'omega', 'real', 'imag'.
    """
    w = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), n_points)
    w_out, H = signal.freqresp(sys, w)

    return {
        'omega': w_out,
        'real': np.real(H),
        'imag': np.imag(H),
    }


def step_response_data(
    sys: signal.TransferFunction,
    t_final: float = 20.0,
    n_points: int = 2000,
) -> Dict[str, np.ndarray]:
    """Compute step response of a transfer function.

    Returns:
        Dict with 'time' (s), 'response', plus transient metrics:
        'rise_time', 'settling_time', 'overshoot', 'steady_state'.
    """
    t = np.linspace(0, t_final, n_points)
    t_out, y = signal.step(sys, T=t)

    # Transient metrics
    ss = y[-1] if len(y) > 0 else 0.0
    if abs(ss) > 1e-10:
        y_norm = y / ss
        # Rise time: 10% to 90%
        rise_idx = np.where((y_norm >= 0.1) & (y_norm <= 0.9))[0]
        rise_time = (t_out[rise_idx[-1]] - t_out[rise_idx[0]]) if len(rise_idx) > 1 else 0.0

        # Overshoot
        overshoot = (np.max(y_norm) - 1.0) * 100.0 if np.max(y_norm) > 1.0 else 0.0

        # Settling time (2% band)
        settled = np.abs(y_norm - 1.0) < 0.02
        if np.any(settled):
            # Find last index where it goes outside the band
            outside = np.where(~settled)[0]
            settling_time = t_out[outside[-1]] if len(outside) > 0 else 0.0
        else:
            settling_time = t_final
    else:
        rise_time = 0.0
        overshoot = 0.0
        settling_time = t_final

    return {
        'time': t_out,
        'response': y,
        'rise_time': float(rise_time),
        'settling_time': float(settling_time),
        'overshoot_pct': float(overshoot),
        'steady_state': float(ss),
    }
