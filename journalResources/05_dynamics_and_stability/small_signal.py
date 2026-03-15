"""
Small-Signal Stability Analysis for DEMS SuperGrid

Provides:
- Numerical linearisation (Jacobian) of the full electromechanical system
- Eigenvalue analysis and mode classification
- Participation factor computation
- Damping ratio and natural frequency extraction
- Modal analysis for inter-area oscillation identification

References:
    P. Kundur, "Power System Stability and Control", McGraw-Hill, 1994.
    IEEE Std 421.5-2016: Recommended Practice for Excitation System Models
    IEEE/CIGRE Joint Task Force, "Definition and Classification of Power
    System Stability", IEEE Trans. Power Systems, 2004.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class OscillationMode:
    """Identified oscillation mode from eigenvalue analysis."""
    eigenvalue: complex
    frequency_hz: float
    damping_ratio: float
    category: str              # 'local_plant', 'inter_area', 'control', 'overdamped'
    participating_generators: List[str]   # Generators most involved
    participation_factors: Dict[str, float]
    dominant_state: str        # Which state variable dominates

    @property
    def stable(self) -> bool:
        return self.eigenvalue.real < 0

    @property
    def critically_damped(self) -> bool:
        return self.damping_ratio >= 0.05


@dataclass
class SmallSignalResult:
    """Complete result from small-signal stability analysis."""
    A_matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    state_names: List[str]
    modes: List[OscillationMode]
    participation_matrix: np.ndarray

    # Summary statistics
    n_unstable: int = 0
    n_poorly_damped: int = 0  # ζ < 0.05
    min_damping_ratio: float = 0.0
    dominant_inter_area_freq_hz: float = 0.0

    @property
    def stable(self) -> bool:
        return self.n_unstable == 0


class SmallSignalAnalyzer:
    """
    Performs small-signal stability analysis on the DEMS SuperGrid.

    Numerically linearises the coupled electromechanical system by applying
    small perturbations to each state variable and measuring the response.
    The resulting A-matrix eigenvalues determine small-signal stability.

    State vector per generator:
        [δ, ω, Eq', Efd, Vr, Vf, Pg, Pm, PSS_w, PSS_ll1, PSS_ll2]

    For N generators, total state dimension = 11 × N.
    """

    def __init__(self, perturbation: float = 1e-5):
        """
        Args:
            perturbation: Finite-difference perturbation size for Jacobian.
        """
        self.eps = perturbation

    def analyze(
        self,
        dynamics_coordinator,
        system_frequency_hz: float = 50.0,
    ) -> SmallSignalResult:
        """Run full small-signal stability analysis.

        Args:
            dynamics_coordinator: DynamicsCoordinator from SuperGrid.
            system_frequency_hz: Current system frequency.

        Returns:
            SmallSignalResult with eigenvalues, modes, and participation factors.
        """
        # 1. Extract state vector and state names
        x0, state_names, gen_ids = self._extract_state_vector(dynamics_coordinator)
        n = len(x0)

        if n == 0:
            logger.warning("No dynamic states found — cannot perform small-signal analysis")
            return SmallSignalResult(
                A_matrix=np.array([[]]),
                eigenvalues=np.array([]),
                eigenvectors=np.array([[]]),
                state_names=[],
                modes=[],
                participation_matrix=np.array([[]]),
            )

        # 2. Compute A-matrix (numerical Jacobian)
        A = self._compute_jacobian(dynamics_coordinator, x0, state_names, gen_ids)

        # 3. Eigenvalue decomposition
        eigenvalues, eigenvectors = linalg.eig(A)

        # 4. Participation factors
        P = self._participation_factors(eigenvectors)

        # 5. Classify modes
        modes = self._classify_modes(eigenvalues, eigenvectors, P,
                                      state_names, gen_ids)

        # 6. Summary stats
        n_unstable = sum(1 for ev in eigenvalues if ev.real > 0)
        damping_ratios = [m.damping_ratio for m in modes if m.frequency_hz > 0.01]
        n_poorly_damped = sum(1 for dr in damping_ratios if dr < 0.05)
        min_dr = min(damping_ratios) if damping_ratios else 1.0

        inter_area = [m for m in modes
                      if m.category == 'inter_area' and m.frequency_hz > 0]
        dominant_ia = inter_area[0].frequency_hz if inter_area else 0.0

        result = SmallSignalResult(
            A_matrix=A,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            state_names=state_names,
            modes=modes,
            participation_matrix=P,
            n_unstable=n_unstable,
            n_poorly_damped=n_poorly_damped,
            min_damping_ratio=min_dr,
            dominant_inter_area_freq_hz=dominant_ia,
        )

        logger.info(
            f"Small-signal analysis: {n} states, "
            f"{n_unstable} unstable, {n_poorly_damped} poorly damped, "
            f"min ζ={min_dr:.4f}"
        )

        return result

    def analyze_from_params(
        self,
        gen_params: List[Dict],
    ) -> SmallSignalResult:
        """Run small-signal analysis from parameter dictionaries.

        This is an alternative entry point when the full DynamicsCoordinator
        is not available (e.g., for testing or offline analysis).

        Args:
            gen_params: List of dicts with keys:
                H, D, Xd_prime, Td0_prime, KA, TA, KE, TE, KF, TF,
                R, TG, TT, Dt, K_pss, T_washout, T_lead1, T_lag1,
                P_mech_pu, P_elec_pu, Vt, delta_rad, Eq_prime, omega
        """
        n_gen = len(gen_params)
        states_per_gen = 11
        n = n_gen * states_per_gen

        state_names = []
        gen_ids = []
        x0 = np.zeros(n)

        for i, gp in enumerate(gen_params):
            gen_id = gp.get('gen_id', f'G{i}')
            gen_ids.append(gen_id)
            base = i * states_per_gen
            state_names.extend([
                f'{gen_id}_delta', f'{gen_id}_omega', f'{gen_id}_Eq_prime',
                f'{gen_id}_Efd', f'{gen_id}_Vr', f'{gen_id}_Vf',
                f'{gen_id}_Pg', f'{gen_id}_Pm',
                f'{gen_id}_PSS_w', f'{gen_id}_PSS_ll1', f'{gen_id}_PSS_ll2',
            ])
            x0[base] = gp.get('delta_rad', 0.0)
            x0[base + 1] = gp.get('omega', 1.0)
            x0[base + 2] = gp.get('Eq_prime', 1.0)
            x0[base + 3] = gp.get('Efd', 1.0)
            x0[base + 4] = gp.get('Vr', 1.0)
            x0[base + 5] = gp.get('Vf', 0.0)
            x0[base + 6] = gp.get('Pg', 1.0)
            x0[base + 7] = gp.get('Pm', 1.0)
            x0[base + 8] = 0.0  # PSS washout
            x0[base + 9] = 0.0  # PSS lead-lag 1
            x0[base + 10] = 0.0  # PSS lead-lag 2

        # Build A-matrix analytically from linearised equations
        A = self._build_analytical_A(gen_params, x0)

        eigenvalues, eigenvectors = linalg.eig(A)
        P = self._participation_factors(eigenvectors)
        modes = self._classify_modes(eigenvalues, eigenvectors, P,
                                      state_names, gen_ids)

        n_unstable = sum(1 for ev in eigenvalues if ev.real > 0)
        damping_ratios = [m.damping_ratio for m in modes if m.frequency_hz > 0.01]
        n_poorly_damped = sum(1 for dr in damping_ratios if dr < 0.05)
        min_dr = min(damping_ratios) if damping_ratios else 1.0

        inter_area = [m for m in modes if m.category == 'inter_area']
        dominant_ia = inter_area[0].frequency_hz if inter_area else 0.0

        return SmallSignalResult(
            A_matrix=A,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            state_names=state_names,
            modes=modes,
            participation_matrix=P,
            n_unstable=n_unstable,
            n_poorly_damped=n_poorly_damped,
            min_damping_ratio=min_dr,
            dominant_inter_area_freq_hz=dominant_ia,
        )

    # ─── Internal Methods ──────────────────────────────────────────

    def _extract_state_vector(
        self,
        dynamics_coordinator,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract the full state vector from a DynamicsCoordinator."""
        state_names = []
        gen_ids = []
        states = []

        if not hasattr(dynamics_coordinator, 'generators'):
            return np.array([]), [], []

        for gen_id, gen in dynamics_coordinator.generators.items():
            gen_ids.append(gen_id)
            state_names.extend([
                f'{gen_id}_delta', f'{gen_id}_omega', f'{gen_id}_Eq_prime',
                f'{gen_id}_Efd', f'{gen_id}_Vr', f'{gen_id}_Vf',
                f'{gen_id}_Pg', f'{gen_id}_Pm',
                f'{gen_id}_PSS_w', f'{gen_id}_PSS_ll1', f'{gen_id}_PSS_ll2',
            ])
            states.extend([
                gen.delta,
                gen.omega,
                gen.Eq_prime,
                gen.exciter.Efd if gen.exciter else 1.0,
                gen.exciter.Vr if gen.exciter else 1.0,
                gen.exciter.Vf if gen.exciter else 0.0,
                gen.governor.Pg if gen.governor else 1.0,
                gen.governor.Pm if gen.governor else 1.0,
                gen.pss.washout_state if gen.pss else 0.0,
                gen.pss.lead_lag1_state if gen.pss else 0.0,
                gen.pss.lead_lag2_state if gen.pss else 0.0,
            ])

        return np.array(states), state_names, gen_ids

    def _compute_jacobian(
        self,
        dynamics_coordinator,
        x0: np.ndarray,
        state_names: List[str],
        gen_ids: List[str],
    ) -> np.ndarray:
        """Compute the system A-matrix using finite-difference Jacobian.

        For each state x_i, perturb by ±ε and compute:
            A[:, i] ≈ (f(x0 + εei) - f(x0 - εei)) / (2ε)
        """
        n = len(x0)
        A = np.zeros((n, n))
        dt_test = 0.001  # Very small time step for derivative approximation

        f0 = self._evaluate_dynamics(dynamics_coordinator, x0, gen_ids, dt_test)

        for i in range(n):
            x_plus = x0.copy()
            x_plus[i] += self.eps
            self._set_state_vector(dynamics_coordinator, x_plus, gen_ids)
            f_plus = self._evaluate_dynamics(dynamics_coordinator, x_plus, gen_ids, dt_test)

            x_minus = x0.copy()
            x_minus[i] -= self.eps
            self._set_state_vector(dynamics_coordinator, x_minus, gen_ids)
            f_minus = self._evaluate_dynamics(dynamics_coordinator, x_minus, gen_ids, dt_test)

            A[:, i] = (f_plus - f_minus) / (2.0 * self.eps)

        # Restore original state
        self._set_state_vector(dynamics_coordinator, x0, gen_ids)

        return A

    def _evaluate_dynamics(
        self,
        dynamics_coordinator,
        x: np.ndarray,
        gen_ids: List[str],
        dt: float,
    ) -> np.ndarray:
        """Evaluate dx/dt at state x by running one micro-step."""
        self._set_state_vector(dynamics_coordinator, x, gen_ids)

        # Snapshot state before
        x_before = x.copy()

        # Run one dynamics step (this modifies the coordinator's internal state)
        if hasattr(dynamics_coordinator, 'step'):
            dynamics_coordinator.step(dt)

        # Extract state after
        x_after, _, _ = self._extract_state_vector(dynamics_coordinator)

        # Restore to original
        self._set_state_vector(dynamics_coordinator, x_before, gen_ids)

        # Approximate dx/dt
        return (x_after - x_before) / dt

    def _set_state_vector(
        self,
        dynamics_coordinator,
        x: np.ndarray,
        gen_ids: List[str],
    ) -> None:
        """Push a state vector back into the DynamicsCoordinator."""
        if not hasattr(dynamics_coordinator, 'generators'):
            return

        idx = 0
        for gen_id in gen_ids:
            gen = dynamics_coordinator.generators.get(gen_id)
            if gen is None:
                idx += 11
                continue

            gen.delta = x[idx]
            gen.omega = x[idx + 1]
            gen.Eq_prime = x[idx + 2]
            if gen.exciter:
                gen.exciter.Efd = x[idx + 3]
                gen.exciter.Vr = x[idx + 4]
                gen.exciter.Vf = x[idx + 5]
            if gen.governor:
                gen.governor.Pg = x[idx + 6]
                gen.governor.Pm = x[idx + 7]
            if gen.pss:
                gen.pss.washout_state = x[idx + 8]
                gen.pss.lead_lag1_state = x[idx + 9]
                gen.pss.lead_lag2_state = x[idx + 10]
            idx += 11

    def _build_analytical_A(
        self,
        gen_params: List[Dict],
        x0: np.ndarray,
    ) -> np.ndarray:
        """Build linearised A-matrix analytically from parameter dicts.

        Uses the analytical Jacobian of the standard electromechanical model
        for each generator. Cross-coupling between generators through the
        network is approximated by the admittance matrix (or ignored for SMIB).
        """
        n_gen = len(gen_params)
        spg = 11  # states per generator
        n = n_gen * spg
        A = np.zeros((n, n))

        for i, gp in enumerate(gen_params):
            base = i * spg
            H = gp.get('H', 3.5)
            D = gp.get('D', 2.0)
            f0 = gp.get('frequency', 50.0)
            omega_base = 2.0 * np.pi * f0
            Xd_prime = gp.get('Xd_prime', 0.3)
            Td0_prime = gp.get('Td0_prime', 8.0)
            KA = gp.get('KA', 200.0)
            TA = gp.get('TA', 0.02)
            KE = gp.get('KE', 1.0)
            TE = gp.get('TE', 0.5)
            KF = gp.get('KF', 0.03)
            TF = gp.get('TF', 1.0)
            R = gp.get('R', 0.05)
            TG = gp.get('TG', 0.2)
            TT = gp.get('TT', 0.5)
            Dt_gov = gp.get('Dt', 0.05)
            K_pss = gp.get('K_pss', 5.0)
            Tw = gp.get('T_washout', 1.41)
            Tld1 = gp.get('T_lead1', 0.154)
            Tlg1 = gp.get('T_lag1', 0.033)
            Tld2 = gp.get('T_lead2', 0.154)
            Tlg2 = gp.get('T_lag2', 0.033)

            Vt = gp.get('Vt', 1.0)
            P_mech_pu = gp.get('P_mech_pu', 0.5)
            P_elec_pu = gp.get('P_elec_pu', 0.5)
            MVA = gp.get('MVA_base', 100.0)

            # Swing equation: dδ/dt = ω_base × (ω - 1)
            #                  dω/dt = (Pm - Pe - D(ω-1)) / (2H)
            # δ column
            A[base, base + 1] = omega_base       # dδ/dω = ω_base
            # ω row
            A[base + 1, base + 1] = -D / (2.0 * H)  # dω/dω
            A[base + 1, base + 7] = 1.0 / (2.0 * H)  # dω/dPm (Pm from governor)

            # Approximate dPe/dδ for SMIB: Pe ≈ Eq'Vt sin(δ)/Xd'
            delta0 = x0[base]
            Eq0 = x0[base + 2]
            if Xd_prime > 0:
                dPe_ddelta = Eq0 * Vt * np.cos(delta0) / Xd_prime
                dPe_dEq = Vt * np.sin(delta0) / Xd_prime
            else:
                dPe_ddelta = 0.0
                dPe_dEq = 0.0

            A[base + 1, base] = -dPe_ddelta / (2.0 * H)    # dω/dδ
            A[base + 1, base + 2] = -dPe_dEq / (2.0 * H)   # dω/dEq'

            # Field flux: dEq'/dt = (Efd - Eq') / Td0'
            A[base + 2, base + 2] = -1.0 / Td0_prime         # dEq'/dEq'
            A[base + 2, base + 3] = 1.0 / Td0_prime          # dEq'/dEfd

            # Exciter (AVR): dEfd/dt ≈ (Vr - KE*Efd) / TE
            A[base + 3, base + 3] = -KE / TE                  # dEfd/dEfd
            A[base + 3, base + 4] = 1.0 / TE                  # dEfd/dVr

            # Regulator: dVr/dt ≈ (KA*(Vref - Vt - Vfb + Vpss) - Vr) / TA
            A[base + 4, base + 4] = -1.0 / TA                 # dVr/dVr
            A[base + 4, base + 5] = KA / TA                   # dVr/dVf (feedback)

            # PSS output feeds into Vr through KA
            A[base + 4, base + 10] = KA / TA                  # dVr/dPSS_ll2

            # Stabilizing feedback: dVf/dt = ((KF/TF)*Efd - Vf) / TF
            A[base + 5, base + 3] = KF / (TF * TF) if TF > 0 else 0.0
            A[base + 5, base + 5] = -1.0 / TF if TF > 0 else 0.0

            # Governor: dPg/dt = (Pref - (ω-1)/R - Pg) / TG
            A[base + 6, base + 1] = -1.0 / (R * TG) if R > 0 else 0.0  # dPg/dω
            A[base + 6, base + 6] = -1.0 / TG                            # dPg/dPg

            # Turbine: dPm/dt = (Pg - Pm) / TT  (+damping: Pm_out = Pm - Dt*(ω-1))
            A[base + 7, base + 6] = 1.0 / TT                   # dPm/dPg
            A[base + 7, base + 7] = -1.0 / TT                  # dPm/dPm

            # PSS washout: dW/dt = (Δω - W) / Tw
            A[base + 8, base + 1] = 1.0 / Tw if Tw > 0 else 0.0   # dW/dω
            A[base + 8, base + 8] = -1.0 / Tw if Tw > 0 else 0.0  # dW/dW

            # PSS lead-lag 1: dLL1/dt = (K_pss*(Δω - W) - LL1) / Tlg1
            A[base + 9, base + 1] = K_pss / Tlg1 if Tlg1 > 0 else 0.0
            A[base + 9, base + 8] = -K_pss / Tlg1 if Tlg1 > 0 else 0.0
            A[base + 9, base + 9] = -1.0 / Tlg1 if Tlg1 > 0 else 0.0

            # PSS lead-lag 2: dLL2/dt = (LL1_out - LL2) / Tlg2
            A[base + 10, base + 9] = 1.0 / Tlg2 if Tlg2 > 0 else 0.0
            A[base + 10, base + 10] = -1.0 / Tlg2 if Tlg2 > 0 else 0.0

        return A

    def _participation_factors(self, V: np.ndarray) -> np.ndarray:
        """Compute participation factor matrix.

        P(i,k) = |V(i,k)| × |W(k,i)| where W = V^{-1} (left eigenvectors).
        P(i,k) measures how much state i participates in mode k.
        Each column is normalised to sum to 1.
        """
        n = V.shape[0]
        if n == 0:
            return np.array([[]])

        try:
            W = linalg.inv(V)
        except linalg.LinAlgError:
            W = linalg.pinv(V)

        P = np.abs(V) * np.abs(W.T)

        # Normalise columns (modes)
        col_sums = P.sum(axis=0)
        col_sums[col_sums < 1e-20] = 1.0
        P = P / col_sums

        return P

    def _classify_modes(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        P: np.ndarray,
        state_names: List[str],
        gen_ids: List[str],
    ) -> List[OscillationMode]:
        """Classify each eigenvalue into oscillation mode categories."""
        modes = []
        n_modes = len(eigenvalues)

        for k in range(n_modes):
            ev = eigenvalues[k]

            # Skip conjugate pairs (keep only positive imaginary)
            if ev.imag < -1e-10:
                continue

            # Frequency and damping
            freq_hz = abs(ev.imag) / (2.0 * np.pi)
            mag = abs(ev)
            if mag > 1e-10:
                damping = -ev.real / mag
            else:
                damping = 1.0

            # Participation: find dominant state and generators
            if P.shape[1] > k:
                pk = P[:, k]
                dominant_idx = np.argmax(pk)
                dominant_state = state_names[dominant_idx] if dominant_idx < len(state_names) else "unknown"

                # Find participating generators
                spg = 11  # states per generator
                gen_participations = {}
                for gi, gid in enumerate(gen_ids):
                    gen_p = pk[gi * spg:(gi + 1) * spg].sum()
                    if gen_p > 0.05:  # >5% participation
                        gen_participations[gid] = float(gen_p)
            else:
                dominant_state = "unknown"
                gen_participations = {}

            # Classify category
            if freq_hz < 0.01:
                if ev.imag == 0:
                    category = 'overdamped'
                else:
                    category = 'control'
            elif 0.1 <= freq_hz <= 1.0:
                # Inter-area if multiple generators from different areas
                areas_involved = set()
                for gid in gen_participations:
                    if '_A_' in gid or gid.startswith('A'):
                        areas_involved.add('A')
                    elif '_B_' in gid or gid.startswith('B'):
                        areas_involved.add('B')
                    elif '_C_' in gid or gid.startswith('C'):
                        areas_involved.add('C')
                category = 'inter_area' if len(areas_involved) > 1 else 'local_plant'
            elif 1.0 < freq_hz <= 3.0:
                category = 'local_plant'
            else:
                category = 'control'

            modes.append(OscillationMode(
                eigenvalue=ev,
                frequency_hz=freq_hz,
                damping_ratio=float(damping),
                category=category,
                participating_generators=list(gen_participations.keys()),
                participation_factors=gen_participations,
                dominant_state=dominant_state,
            ))

        # Sort by frequency
        modes.sort(key=lambda m: m.frequency_hz)
        return modes


def quick_eigenvalue_analysis(
    generators: List[Dict],
) -> Dict[str, object]:
    """Quick eigenvalue analysis from generator parameter dictionaries.

    Convenience function for the evaluation pipeline.

    Args:
        generators: List of dicts with generator dynamic parameters.

    Returns:
        Dict with 'result' (SmallSignalResult), 'eigenvalues', 'modes'.
    """
    analyzer = SmallSignalAnalyzer()
    result = analyzer.analyze_from_params(generators)

    return {
        'result': result,
        'eigenvalues': result.eigenvalues,
        'modes': result.modes,
        'n_unstable': result.n_unstable,
        'n_poorly_damped': result.n_poorly_damped,
        'min_damping': result.min_damping_ratio,
        'stable': result.stable,
    }
