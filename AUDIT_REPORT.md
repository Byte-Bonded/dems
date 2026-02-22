# DEMS Microgrid Simulation — Comprehensive Technical Audit

**Date:** 2025  
**Scope:** Full codebase under `src/simulation/`, `src/core/`, `src/`, `tests/`  
**Lines reviewed:** ~8 500 across 15+ source files  
**Standards referenced:** IEEE Std 39-bus, IEEE Std 421.5 (IEEET1), IEEE/NERC TGOV1, IEEE Std 421.5 PSS2A, IEGC (Indian Grid Code), IEC 61400-27, IEEE Std C37.106, IEEE Std 1547

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Top 10 Critical Issues](#2-top-10-critical-issues)
3. [IEEE Standard Compliance Violations](#3-ieee-standard-compliance-violations)
4. [Bug Catalogue by Category](#4-bug-catalogue-by-category)
5. [Numerical Stability Report](#5-numerical-stability-report)
6. [Architecture & Design Flaws](#6-architecture--design-flaws)
7. [Test Coverage Gaps](#7-test-coverage-gaps)
8. [Systemic Risk Summary](#8-systemic-risk-summary)
9. [Refactoring Recommendations](#9-refactoring-recommendations)
10. [Appendix — Full Bug Index](#10-appendix--full-bug-index)

---

## 1. Executive Summary

The DEMS codebase implements a Tri-Area 117-bus supergrid (3 × IEEE 39-bus New England) with dynamic models, DER management, and an RL orchestrator. The simulation broadly covers the right scope (swing dynamics, AVR, governor, PSS, AGC, ZIP loads, protection) but contains **5 CRITICAL**, **14 HIGH**, **16 MEDIUM**, and **8 LOW** severity bugs, many of which stem from **incorrect or incomplete implementation of IEEE standards**.

**Key findings:**
- The IEEE IEEET1 exciter, TGOV1 governor, and PSS2A stabilizer all deviate from their reference block diagrams in structurally important ways.
- AGC computes adjustments but never applies them to generator mechanical power.
- The 117-bus topology contains an out-of-range bus reference (bus 39 does not exist in 0-indexed IEEE 39-bus).
- Generator dynamics are not properly initialized from power flow steady state.
- Forward Euler integration with 10 ms step is marginally stable for stiff AVR dynamics.
- Solar PV output is physically impossible (1000 m² panel area for a 50 MW farm).
- The dynamics simulation covers only 0.1 s of each 300 s control step (0.03%).

---

## 2. Top 10 Critical Issues

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | **CRITICAL** | `dynamics.py` L510-520 | AGC `update()` returns adjustments dict but `DynamicsCoordinator.step()` **never applies them** to governor `Pref` — secondary frequency control is dead code |
| 2 | **CRITICAL** | `dynamics.py` L152-160 | `Eq_prime` initialization ignores rotor angle: uses `√(Vt² + (Xd'·I)²)` instead of proper phasor formula `|Vt + jXd'·I|` — causes incorrect initial internal EMF |
| 3 | **CRITICAL** | `supergrid.py` L170 | Tie-line `TL_AB_3` uses `from_bus_local=39` — IEEE 39-bus has buses 0–38, so bus 39 is **out of range**, creating a line to a potentially non-existent bus |
| 4 | **CRITICAL** | `supergrid.py` L637-642 | `_configure_slack_bus()` removes all ext_grid entries except the first — discards ~2000 MVA of slack generation from Areas B and C, drastically unbalancing the merged grid |
| 5 | **CRITICAL** | `der.py` L211-216 + `supergrid.py` L507 | Solar PV: `panel_area_m2=1000` for a 50 MW farm; at 1000 W/m² × 20% efficiency → max 0.2 MW. Actual output will be **250× below** rated capacity |
| 6 | **HIGH** | `dynamics.py` (entire AVR) | IEEET1 excitation system missing saturation function SE(Efd), missing KF feedback loop — violates IEEE Std 421.5 §5.1 block diagram |
| 7 | **HIGH** | `dynamics.py` L339-345 | Governor/exciter states initialized to hard 1.0 pu regardless of power flow steady state — first dynamics step causes transient kickback |
| 8 | **HIGH** | `orchestrator.py` L499-501 | Wind curtailment applied to wind speed (cubic relation) not to power — 20% speed reduction = 49% power reduction. Solar has the same issue with non-linear irradiance-to-power mapping |
| 9 | **HIGH** | `orchestrator.py` L758-760 | Load scale/restore uses `scale × (1/scale)` without storing original values — floating-point error accumulates every step, drifting loads irreversibly |
| 10 | **HIGH** | `orchestrator.py` L739, L741 | Dynamics substeps: `10 × 0.01s = 0.1s` out of a 300s control step — **99.97% of inter-step time** is not simulated; frequency dynamics are physically meaningless |

---

## 3. IEEE Standard Compliance Violations

### 3.1. IEEE Std 421.5 — Type 1 Excitation System (IEEET1)

**Reference:** IEEE Std 421.5-2016, §5.1, Figure 5-1

| ID | Violation | Detail | Impact |
|----|-----------|--------|--------|
| IEEE-01 | **Missing saturation function SE(Efd)** | The IEEET1 block diagram requires a saturation feedback `SE(Efd)·Efd` subtracted inside the exciter block. The code (`dynamics.py` L303-310) uses a simple first-order `dEfd = (KE·Vr - Efd) / TE · dt` with no saturation term. | Over-predicts Efd at high excitation; removes the dominant non-linearity that limits generator reactive power capability. |
| IEEE-02 | **Missing KF stabilizing feedback** | IEEE IEEET1 has a derivative feedback path `KF/sTF` from Efd back to the regulator summing junction. This is completely absent. | Reduces phase margin; the AVR will oscillate more readily than the real system, giving an optimistic voltage stability picture. |
| IEEE-03 | **Incorrect exciter block structure** | IEEE IEEET1 exciter block is `1/(KE + sTE)` with internal feedback. Code implements `(KE·Vr - Efd)/TE` which is a different transfer function: `KE/(1 + sTE)` instead of `1/(KE + sTE)`. When `KE ≠ 1` these are not equivalent. | Gain and time constant of the exciter are both wrong for any `KE ≠ 1`. |
| IEEE-04 | **Steady-state Efd not initialized from PF** | `ExcitationSystem.__init__` sets `Efd = 1.0 pu` and `Vr = 1.0 pu` regardless of power flow solution. IEEE practice is to solve the AVR states from the PF operating point so the first dynamics step has zero mismatch. | First step jumps: the regulator output required to hold the PF voltage is NOT 1.0 pu for most generators, so immediate transients appear. |
| IEEE-05 | **Vref not initialized from PF terminal voltage** | `Vref` defaults to 1.0 pu in `ExciterParams`. Should be set to the PF terminal voltage of each generator so the AVR error is zero at t=0. | Every generator with Vt ≠ 1.0 will see a non-zero AVR error and start regulating from the first timestep, producing a spurious transient. |

### 3.2. IEEE/NERC TGOV1 — Governor-Turbine Model

**Reference:** NERC Power Plant Model Verification (PPMV) guidelines; IEEE PES Task Force on Turbine-Governor Modeling

| ID | Violation | Detail | Impact |
|----|-----------|--------|--------|
| IEEE-06 | **Missing valve position limits with rate limiting** | TGOV1 reference model has a rate-limited valve (`VMAX/VMIN` with slew rate). Code only clips Pg to `Pmin/Pmax` after integration — no rate limit on valve movement. | Governor responds too fast to frequency events; over-estimates primary frequency response capability. |
| IEEE-07 | **Missing damping term (Dt)** | Standard TGOV1 has a turbine damping coefficient Dt that creates a direct path from speed to mechanical power. Code omits this. | Under-estimates mechanical damping, making frequency nadir analysis slightly pessimistic. |
| IEEE-08 | **Governor Pref/Pm not initialized from PF** | `GovernorTurbine.__init__` sets `Pg = 1.0 pu` and `Pm = 1.0 pu`. Should be `Pm = P_elec_pu` from power flow. | If a generator is dispatched at 0.6 pu, the governor starts at 1.0 pu and immediately tries to produce 0.4 pu more power than needed — massive frequency excursion on first step. |
| IEEE-09 | **Droop applied to absolute speed, not incremental** | Code: `Pg_ref = Pref - delta_omega / R`. The standard formulation is `Pg_ref = Pref - (omega - omega_ref) / R` where `omega_ref` can be adjusted. Minor but affects multi-machine droop coordination. | Limited impact when omega_ref = 1.0 (which it is), but prevents future use of intentional speed bias for AGC coordination. |

### 3.3. IEEE Std 421.5 PSS2A — Power System Stabilizer

**Reference:** IEEE Std 421.5-2016, §8.2, Figure 8-2

| ID | Violation | Detail | Impact |
|----|-----------|--------|--------|
| IEEE-10 | **PSS2A requires dual-input (speed + power)** | IEEE PSS2A is a dual-input stabilizer: input 1 = speed signal through ramp tracking filter, input 2 = power signal. Code uses only speed deviation as a single input. | This is actually a PSS1A variant (single-input), not PSS2A. Mislabeled and missing the power input channel that provides improved damping at lower frequencies. |
| IEEE-11 | **Missing ramp tracking filter** | PSS2A uses ramp tracking filters (M = 2 or 5 stages) to extract the speed signal without steady-state offset. Code uses a simple washout instead. | Ramp changes in frequency (from AGC) will bleed through the washout and produce false PSS output. |
| IEEE-12 | **Lead-lag compensator missing feed-through term** | A standard 1st-order lead-lag is `(1 + sT_lead) / (1 + sT_lag)`. At DC the gain is 1.0 (feed-through). The code state update `d_lead_lag = (input·(T_lead/T_lag) - state) / T_lag · dt` sets DC gain to `T_lead/T_lag` = 4.67 instead of 1.0. | PSS output is amplified by 4.67× at low frequencies, causing excessive AVR modulation and potential oscillatory instability. |
| IEEE-13 | **Only one lead-lag stage** | IEEE PSS2A specifies two cascaded lead-lag stages (T1/T2 and T3/T4). Code has only one. | Insufficient phase compensation; PSS cannot provide adequate damping for both local and inter-area oscillation modes simultaneously. |
| IEEE-14 | **Gain K_PSS = 9.5 may be excessive** | Typical PSS gains are 1–20; 9.5 combined with the 4.67× DC gain error gives effective gain ~44, which is dangerously high. | Risk of PSS-induced oscillations or AVR / PSS hunting (sustained limit-cycle oscillations). |

### 3.4. IEEE 39-Bus (New England Test System) Topology

**Reference:** IEEE 39-bus test case (Athay et al., 1979; Pai, 1989)

| ID | Violation | Detail | Impact |
|----|-----------|--------|--------|
| IEEE-15 | **Bus indexing inconsistency (0 vs 1)** | The IEEE 39-bus standard numbers buses 1–39. pandapower's `case39()` returns buses 0–38 (0-indexed). `IEEE39_GENERATOR_DATA` maps buses 30–39 (1-indexed). Tie-line `TL_AB_3` uses `from_bus_local=39` which is bus 40 in the standard and out-of-range in pandapower. | Tie-line connects to wrong bus or crashes. Generator dynamic models may be assigned to wrong buses (off by one). |
| IEEE-16 | **Generator data uses non-standard H values** | IEEE 39-bus standard (Pai, 1989) gives H values in seconds on machine base. Code's `IEEE39_GENERATOR_DATA` has `H=500` for Gen 1 (bus 30) which represents it as an equivalent aggregated system. This is correct per the standard, BUT the code uses it directly in the swing equation without noting that this effectively makes Gen 1 an infinite bus (any disturbance has negligible effect on this generator). | Gen 1 acts as a second slack bus in dynamics, potentially masking frequency instability. |
| IEEE-17 | **Missing transformer tap ratios** | IEEE 39-bus has 12 transformers with specific tap ratios. After `pp.merge_nets()`, transformer tap ratios may be reset or duplicated. The code doesn't verify transformer data post-merge. | Incorrect voltage transformation ratios affect power flow solution accuracy. |
| IEEE-18 | **Generator MVA bases inconsistent with pandapower** | `IEEE39_GENERATOR_DATA` specifies MVA bases (e.g., Gen 2 = 520 MVA) but pandapower's `case39()` uses `baseMVA=100`. The code doesn't verify that pandapower line/transformer impedances are on the same base as the dynamic model. | Per-unit conversion errors between power flow and dynamic models. |
| IEEE-19 | **Ext_grid removal loses voltage control** | `_configure_slack_bus()` removes ext_grid elements from Areas B and C. In the original IEEE 39-bus, bus 39 hosts the equivalent system (infinite bus). Removing 2 of 3 equivalent systems removes ~2000 MVA of voltage/power support. | Severe voltage and reactive power regulation problems in Areas B and C. Not realistic for a 3-area interconnection. |

### 3.5. Indian Grid Code (IEGC) Compliance

**Reference:** CERC (Indian Electricity Grid Code) Regulations, 2010 (as amended)

| ID | Violation | Detail | Impact |
|----|-----------|--------|--------|
| IEEE-20 | **UFLS scheme incomplete** | IEGC mandates staged Under-Frequency Load Shedding (UFLS) at multiple thresholds (e.g., 49.2, 49.0, 48.8 Hz) with specific percentages. The `ProtectionRelay` only has a single under-frequency trip at 49.5 Hz with a flat delay. | No staged load shedding means the simulation either trips everything at 49.5 Hz or nothing. Real grids use 3-5 UFLS stages to arrest frequency decline. |
| IEEE-21 | **OFGT scheme missing** | IEGC specifies Over-Frequency Generator Tripping (OFGT) at multiple stages. `ProtectionRelay` has a single over-frequency trip at 50.5 Hz. | Same problem as UFLS — no staged response. |
| IEEE-22 | **Frequency deadband inconsistent with IEGC** | AGC deadband is set to 0.02 Hz. IEGC specifies a secondary control deadband of ±0.03 Hz (CERC regulation). | Minor but results in AGC acting when it should be quiescent. |
| IEEE-23 | **Protection trip delay for frequency too long** | `frequency_trip_delay = 0.3s` for both UF and OF. IEGC UFLS stage 1 operates at ~100–200 ms. | Under-frequency events persist 50-200% longer than they would in the real grid before load is shed. |
| IEEE-24 | **No islanding detection** | IEGC requires anti-islanding protection for DER. Code has no islanding detection or ride-through logic for DER units. | DER behaviour during grid disturbances is unrealistic. |

### 3.6. IEC 61400-27 — Wind Turbine Model

**Reference:** IEC 61400-27-1:2015 (Wind turbines – Electrical simulation models)

| ID | Violation | Detail | Impact |
|----|-----------|--------|--------|
| IEEE-25 | **Missing cut-out speed in DERManager** | `der.py` `set_wind_output()` has no cut-out speed — above 15 m/s it outputs rated power forever. IEC 61400-27 and real turbines cut out at 25 m/s. The standalone `microgrid.py` `WindTurbine` class correctly implements cut-out at 25 m/s. | Over-predicts wind generation during storms; wind farms would continue producing at hurricane-force winds. |
| IEEE-26 | **Power curve shape incorrect** | Code uses `((v-3)/12)³` which cubes the normalised speed delta. IEC standard power curve is `P ∝ v³` (cubic in absolute speed) between cut-in and rated. Correct: `(v³ - v_ci³)/(v_r³ - v_ci³)`. | Under-predicts output at moderate speeds, over-predicts at high speeds. At 9 m/s: code gives 12.5% capacity factor, actual ~42%. |
| IEEE-27 | **No reactive power capability / LVRT** | IEC 61400-27 requires Low Voltage Ride-Through (LVRT) and reactive current injection during faults. Wind DER `q_mvar=0` always. | During voltage dips, wind turbines disconnect instead of providing grid support as required by modern grid codes. |

### 3.7. IEEE Std 1547 — DER Interconnection

**Reference:** IEEE Std 1547-2018 (Standard for Interconnection of DER)

| ID | Violation | Detail | Impact |
|----|-----------|--------|--------|
| IEEE-28 | **No voltage/frequency ride-through for DER** | IEEE 1547 Category III requires DER to ride through voltage and frequency excursions within defined envelopes. DER models have no ride-through logic — they are either on or off. | During grid disturbances all DER trips immediately instead of providing momentary support, making the simulation overly fragile. |
| IEEE-29 | **No DER volt-var / freq-watt capability** | IEEE 1547-2018 requires DER to provide volt-var and freq-watt response. No DER in the code has autonomous voltage/frequency support. | DER is modeled as passive generation rather than active grid participants as required by modern interconnection standards. |
| IEEE-30 | **Solar PV modeled without inverter dynamics** | IEEE 1547 and IEEE 2800 require utility-scale PV to have grid-forming or grid-following inverter models with current limiting, PLL, etc. Code treats PV as a simple static generator. | Dynamic behaviour during transients is missing; PV response to frequency/voltage events is unrealistic. |

### 3.8. Power Flow Standards (IEEE Std)

| ID | Violation | Detail | Impact |
|----|-----------|--------|--------|
| IEEE-31 | **Losses calculation incorrect** | `power_flow.py` L242-243: `total_losses = total_generation - total_load`. This includes storage, shunt consumption, and slack mismatch — not just I²R losses. Correct: sum `res_line.pl_mw` + `res_trafo.pl_mw`. | Over-reports losses; penalizes RL agent for shunt reactive compensation. |
| IEEE-32 | **Voltage violation thresholds hardcoded** | `power_flow.py` L251: uses `0.95/1.05` hardcoded instead of reading from `SuperGridConfig.v_min_pu / v_max_pu`. | Config changes don't propagate to violation detection. |
| IEEE-33 | **NR solver Jacobian inconsistent with Ybus for tap** | `microgrid.py` standalone NR solver: branch flow formulas use `tap_complex = tap·e^(jΦ)` but Jacobian does not account for off-nominal taps — it uses standard formulas assuming ideal transformers. | Convergence may fail for networks with transformers; if it converges, tap transformer flows will be incorrect. |

---

## 4. Bug Catalogue by Category

### 4.1. Numerical / Integration Bugs

| ID | Severity | File:Line | Description |
|----|----------|-----------|-------------|
| N01 | HIGH | `dynamics.py` L179-183 | Forward Euler integration: `x += dx·dt`. For `TA=0.02s` and `KA=200`, the AVR pole is at `−1/TA = −50`. Euler stability requires `dt < 2/50 = 0.04s`. At `dt=0.01s` the gain margin is only 2.5×. Any parameter reduction in TA pushes this into instability. |
| N02 | HIGH | `dynamics.py` L180-183 | AVR integration: `dVr = (KA·Ve - Vr)/TA · dt`. Factor is `KA/TA = 200/0.02 = 10000 s⁻¹`. `dVr` per step ≈ `10000·Ve·0.01 = 100·Ve`. Even a 1% voltage error produces `dVr = 1.0 pu`, which immediately hits the ±5 pu limiter. AVR is effectively bang-bang, not regulating. |
| N03 | MEDIUM | `dynamics.py` L370-373 | Washout filter: `d_washout = (input - state) / T · dt`. First-order approximation of a high-pass filter. For `T_washout = 1.41s` and `dt = 0.01s`, this is acceptable, but the washout output should be `K_pss · (input - state)`, not stored in `lead_lag_state` directly. The signal path is scrambled. |
| N04 | MEDIUM | `orchestrator.py` L758 | Load scale restore: `scale_loads(area, 1.0/scale)`. If `scale ≈ 1.0`, `1.0/scale ≈ 1 - ε` due to float precision. Over 288 steps, loads drift by up to ~0.01% per step = ~3% cumulative error. |

### 4.2. Initialization / State Bugs

| ID | Severity | File:Line | Description |
|----|----------|-----------|-------------|
| I01 | CRITICAL | `dynamics.py` L152-160 | `Eq_prime = √(Vt² + (Xd'·I)²)` ignores angle between terminal voltage and current. Correct: `E_q' = |V_t∠δ + jX_d'·I∠(δ-φ)|`. The scalar formula over-estimates Eq' for heavily loaded generators. |
| I02 | HIGH | `dynamics.py` L257, L338 | Exciter `Efd=1.0`, `Vr=1.0` and governor `Pg=1.0`, `Pm=1.0` hardcoded. Should be computed from PF: `Efd_0` from generator Q output, `Pm_0 = Pe_0` from power flow P. |
| I03 | HIGH | `supergrid.py` L401-403 | `initialize_dynamics()` calls `gen.initialize(P_mw, Q_mvar, Vt, Va_deg)` which sets `P_mech = P_elec`, but does NOT set governor `Pref` or `Pm`, nor does it set exciter `Efd` to the required steady-state value. |
| I04 | MEDIUM | `der.py` L179-182 | `_battery_states` dict is never populated. `add_battery()` adds to `battery_soc` (legacy dict) but never creates a `BatteryState` object in `_battery_states`. The enhanced SOC-check code in `set_battery_power()` that reads `_battery_states` is dead code. |
| I05 | LOW | `dynamics.py` (post-PF init) | After `supergrid.py` initializes generators from PF, the check `gen_mask = self.net.gen.bus == global_bus` uses the IEEE39 data buses (30-39) + offset. These are 1-indexed but pandapower is 0-indexed, so the mask may match the wrong generators. |

### 4.3. Control Logic Bugs

| ID | Severity | File:Line | Description |
|----|----------|-----------|-------------|
| C01 | CRITICAL | `dynamics.py` L872-876 | `DynamicsCoordinator.step()` calls `agc.update()` and stores adjustments in `agc_adjustments` dict, but **never applies** them. The adjustments are returned in the result dict but no code feeds them back to governor `Pref`. AGC is entirely decorative. |
| C02 | HIGH | `orchestrator.py` L495-505 | Wind curtailment: `effective_ws = ws * (1 - curtail)`. Since wind power ∝ v³, a curtailment of 0.2 (20%) reduces speed to 0.8v, giving power ∝ (0.8v)³ = 0.512·v³ — a 48.8% power reduction. Agent's notion of "20% curtailment" is actually ~49%. Solar has identical issue: power is non-linear in irradiance (via panel physics). |
| C03 | HIGH | `orchestrator.py` L732-735 | `scale_loads(area, scale)` is called for all areas, then after PF and dynamics, `scale_loads(area, 1/scale)` is called. However `scale_loads` multiplies the current `p_mw` values. After DER additions, `area.load_indices` may not include DER loads, so DER load contributions are not scaled back, accumulating bias. |
| C04 | MEDIUM | `orchestrator.py` L279-280 | Per-area DER observation: `obs[base+7] = der_status["solar"]["current_output_mw"] / (3 * 100.0)`. This takes the **total** solar output across all areas and divides by 3. Should be per-area solar output (areas have different DER mixes). |
| C05 | MEDIUM | `dynamics.py` L910 | `create_ieee39_dynamics()` AGC participation factor: `pf = MVA/5000`. Sum across 9 generators (excluding slack): `(520+650+632+508+650+560+540+830)/5000 = 0.978`. Not normalized to 1.0 — AGC under-allocates by 2.2%. |

### 4.4. Unit / Scaling Bugs

| ID | Severity | File:Line | Description |
|----|----------|-----------|-------------|
| U01 | CRITICAL | `der.py` L443-446 | Solar output: `output_mw = (panel_area_m2 · irradiance_kw/m2 · efficiency) / 1000`. For default `panel_area_m2=1000`, `irradiance=1.0 kW/m²`, `efficiency=0.20`: output = `1000·1.0·0.20/1000 = 0.2 MW`. Rated capacity is 50 MW. The formula is correct for a 1000 m² installation; the problem is `panel_area_m2` should be ~250,000 m² for 50 MW. |
| U02 | MEDIUM | `microgrid.py` L272 | `DieselGenerator.ramp_rate = 0.5` — documented as MW/min but `set_output()` uses `max_change = ramp_rate · dt` where `dt` is in hours (1.0 default). So effective ramp = 0.5 MW/hr, not 0.5 MW/min. |
| U03 | MEDIUM | `der.py` L853-856 | `update_battery_soc()`: `delta_soc = -energy_mwh / capacity_mwh`. Doesn't apply round-trip efficiency — discharging 10 MWh at 92% efficiency should remove `10/0.92 = 10.87 MWh` from stored energy. The `BatteryState.update_soc()` class handles this correctly but is never called (see I04). |

### 4.5. Topology / Connectivity Bugs

| ID | Severity | File:Line | Description |
|----|----------|-----------|-------------|
| T01 | CRITICAL | `supergrid.py` L170 | `TL_AB_3: from_bus_local=39`. Bus indices in pandapower `case39()` are 0–38. Bus 39 doesn't exist. pandapower will either create a line to a phantom bus or raise an error at power flow time. |
| T02 | CRITICAL | `supergrid.py` L637-642 | `_configure_slack_bus()` does `self.net.ext_grid = self.net.ext_grid.iloc[:1].copy()` — deletes ext_grids from merged Areas B and C. This removes ~2000 MVA of swing generation. |
| T03 | HIGH | `supergrid.py` L637 | Hardcoded `slack_bus = 30` is 1-indexed (IEEE standard) but pandapower uses 0-indexed. If pandapower `case39()` maps the slack to bus index 29, this assigns ext_grid to the wrong bus. Actually, pandapower does use 0-indexed naming for `case39()`: bus 0 is "Bus 1" in IEEE numbering. The code uses bus 30 meaning it's actually bus 31 in IEEE terms, which is Gen 2, NOT the slack. |
| T04 | MEDIUM | `supergrid.py` L584-588 | `_update_area_mappings()` runs during build but not after DER initialization. DER loads and sgens added later are not in `area.load_indices`, so area-level metrics from `get_area_state()` undercount. |

### 4.6. Power Flow Bugs

| ID | Severity | File:Line | Description |
|----|----------|-----------|-------------|
| P01 | MEDIUM | `power_flow.py` L242-243 | Losses = gen − load includes shunt compensation and storage. Not meaningful as a "loss" metric. Should use `sum(net.res_line.pl_mw) + sum(net.res_trafo.pl_mw)`. |
| P02 | MEDIUM | `power_flow.py` L251 | Hardcoded `0.95/1.05` threshold ignores `SuperGridConfig` values. |
| P03 | LOW | `power_flow.py` L164 | Iteration count: `net.get("_ppc", {}).get("iterations", 0)` — `_ppc` is a pandapower internal that may not be populated in all versions. Returns 0 when unavailable. |
| P04 | LOW | `microgrid.py` Ybus/Jacobian | Standalone NR solver Jacobian does not account for transformer taps in off-diagonal elements — converges for radial feeders but may fail for meshed networks with transformers. |

### 4.7. Thread Safety Bugs

| ID | Severity | File:Line | Description |
|----|----------|-----------|-------------|
| TS01 | HIGH | `der.py` L652-670 | `set_demand_response_curtailment()` is NOT protected by `self._lock`, unlike all other public methods. Concurrent access to this method will cause data races. |
| TS02 | MEDIUM | `der.py` L854 | `update_battery_soc()` acquires no lock. If called concurrently with `set_battery_power()`, SOC can be corrupted. |
| TS03 | LOW | `orchestrator.py` entire | `GridOrchestrator` has no thread safety. If a monitoring thread reads state while `step()` is in progress, it can see partially-updated grid state. |

### 4.8. Architecture / Code Duplication

| ID | Severity | File | Description |
|----|----------|------|-------------|
| A01 | MEDIUM | `microgrid.py` vs `der.py` | Duplicate DER models: `SolarPV`, `WindTurbine`, `BatteryESS` in `microgrid.py` AND `DERManager.set_solar_output()`, `set_wind_output()`, `set_battery_power()` in `der.py`. Different physics models — wind cut-out exists in `microgrid.py` but not in `der.py`. |
| A02 | MEDIUM | `orchestrator.py` vs `src/orchestrator.py` | Two `GridOrchestrator` classes — `src/simulation/orchestrator.py` (RL interface, 885 lines) and `src/orchestrator.py` (monitoring wrapper, 493 lines). Name collision, unclear which is used. |
| A03 | MEDIUM | `grid_manager.py` L375-385 | Deprecated `GridManager` class aliases: `get_current_state()` calls `self.get_system_summary()` which doesn't exist — raises `AttributeError` at runtime. |
| A04 | LOW | `dynamics.py` `create_ieee39_dynamics()` | Creates a standalone `DynamicsCoordinator` but `SuperGrid.initialize_dynamics()` creates its own. The standalone function is dead code (never called from the main simulation path). |
| A05 | LOW | `src/core/energy_manager.py` | References `DERManager` patterns but implements its own simplified version. Unclear relationship to `src/simulation/der.py`. |

### 4.9. Performance Bugs

| ID | Severity | File:Line | Description |
|----|----------|-----------|-------------|
| PERF01 | HIGH | `orchestrator.py` L739 | `dynamics_substeps=10, dt=0.01s` → 0.1s simulated per 300s step. Either increase to `30000` substeps (impractical) or use much larger `dt` with implicit integration. |
| PERF02 | MEDIUM | `microgrid.py` Jacobian | `O(n²)` double-loop Jacobian construction in Python — unusable for systems larger than ~50 buses. Should use sparse matrix operations. |
| PERF03 | LOW | `der.py` `_get_spec()` | Linear scan through `der_specs` list for every query. Should be a dict lookup by name. |

---

## 5. Numerical Stability Report

### 5.1. Forward Euler Stability Analysis

The simulation uses explicit Forward Euler throughout. Stability requires `|1 + λ·dt| < 1` for all eigenvalues λ of the linearized system.

| Model | Fastest Pole (λ) | Max Stable dt | Actual dt | Margin |
|-------|-------------------|---------------|-----------|--------|
| AVR regulator | −KA/TA = −10000 | 0.0002 s | 0.01 s | **UNSTABLE** (50× over limit) |
| AVR transducer | −1/TR = −50 | 0.04 s | 0.01 s | 4× |
| Exciter | −KE/TE = −2 | 1.0 s | 0.01 s | 100× |
| Governor | −1/TG = −5 | 0.4 s | 0.01 s | 40× |
| Turbine | −1/TT = −2 | 1.0 s | 0.01 s | 100× |
| Swing equation | ±j·ωn ≈ ±j3 | N/A (oscillatory) | 0.01 s | OK |

**Critical finding:** The AVR regulator pole at −10000 rad/s is **50× faster** than the Euler stability limit at dt = 0.01 s. The only reason the simulation doesn't blow up is that `Vr` is hard-clipped to [−5, 5] pu on every step, masking the numerical instability with a saturated bang-bang controller.

### 5.2. Recommended Fix

Replace Forward Euler with **Implicit Trapezoidal** (Crank-Nicolson) for all first-order differential equations. For the AVR:

```
Vr_new = Vr + dt/2 · [(KA·Ve_old - Vr)/TA + (KA·Ve_new - Vr_new)/TA]
```

Solving for `Vr_new`:
```
Vr_new = [Vr·(1 - dt/(2·TA)) + dt·KA·Ve/(TA)] / (1 + dt/(2·TA))
```

This is unconditionally stable for any dt.

---

## 6. Architecture & Design Flaws

### 6.1. Missing Power Flow ↔ Dynamics Coupling

The simulation runs dynamics on state variables (`omega`, `delta`, `Eq'`) but these never feed back into the pandapower power flow. The power flow always uses the fixed generator setpoints, not the dynamic states. This means:

- Generator angles from dynamics don't affect power flow
- `Eq_prime` changes from the exciter don't affect reactive power
- Governor power changes don't affect the dispatch

**Impact:** Dynamics and steady-state are completely decoupled. The simulation runs two independent models that don't influence each other.

### 6.2. AGC Signal Path Broken

```
AGC.update() → Dict[gen_id, delta_P_mw]  (computed)
DynamicsCoordinator.step() → agc_adjustments  (stored in result dict)
                           → (nothing applies delta_P to GovernorTurbine.Pref)
```

The return value is available in the `step()` return dict but no consumer reads it and feeds it back.

### 6.3. No Electromechanical → Network Feedback

The dynamics compute generator power changes, but the power flow runs on fixed pandapower `gen.p_mw` values. The standard simulation pipeline should be:

1. Power flow → voltages and currents
2. Calculate Pe for each generator from network
3. Dynamics step updates omega, delta, Efd, Pm
4. Update generator P/Q from dynamics  
5. Re-solve power flow

Steps 4–5 are missing.

---

## 7. Test Coverage Gaps

| ID | Missing Test | Why It Matters |
|----|-------------|----------------|
| TEST01 | No test verifies AGC adjustments are applied to generators | AGC is dead code (C01) — a test would catch this |
| TEST02 | No test checks governor/exciter initial conditions match PF | Initialization bugs (I02, I03) produce transient kickback |
| TEST03 | No test verifies solar output vs capacity at peak irradiance | 0.2 MW from a 50 MW farm (U01) goes undetected |
| TEST04 | No test for wind cut-out speed | Wind generates at hurricane speeds (IEEE-25) |
| TEST05 | No test validates tie-line bus indices against network | Bus 39 out-of-range (T01) would be caught |
| TEST06 | No IEEE 39-bus benchmark validation | No test compares power flow results to published IEEE 39-bus solutions |
| TEST07 | Dynamics test is 20 steps × 0.01s = 0.2s | Far too short to detect oscillatory instability or AGC problems |
| TEST08 | No test for PSS impact on damping | PSS gain bug (IEEE-12, IEEE-14) would be detected |
| TEST09 | No multi-episode RL convergence test | No verify that reward improves across episodes |
| TEST10 | No `set_demand_response_curtailment` thread safety test | Missing lock (TS01) would cause flaky test |

---

## 8. Systemic Risk Summary

| Risk | Likelihood | Impact | Root Cause |
|------|-----------|--------|------------|
| Frequency simulation meaningless | Certain | Complete | AGC dead, dynamics decoupled, 0.03% coverage |
| Voltage profile unrealistic | High | Major | AVR bang-bang, no Efd → Q coupling, ext_grid removal |
| Solar DER negligible output | Certain | High | panel_area not scaled to capacity |
| Wind over-generation in storms | High | Medium | No cut-out speed in DER manager |
| RL agent learns wrong physics | Certain | Critical | All of the above compound |
| Power flow non-convergence | Medium | High | Removing 2 ext_grids, bus 39 tie-line |
| Oscillatory instability reported | Low | Medium | PSS gain too high + wrong DC gain |

---

## 9. Refactoring Recommendations

### Priority 1 — Fix Physics (Blocks Real Progress)

1. **Initialize all dynamic states from power flow.** After `runpp()`, solve for steady-state Efd, Vr, Pm, Pg, and set `omega=1, delta=Va_deg` for each generator.
2. **Apply AGC adjustments.** In `DynamicsCoordinator.step()`, after AGC update, loop through adjustments and add `delta_P` to each governor's `Pref`.
3. **Fix solar panel_area.** Either compute from capacity: `panel_area = capacity_mw * 1e6 / (1000 * efficiency)` or remove the area-based model and use `output = capacity * irradiance / 1000 * efficiency_factor`.
4. **Fix bus indexing.** Map `IEEE39_GENERATOR_DATA` keys to 0-indexed. Change `TL_AB_3.from_bus_local` from 39 to 38.
5. **Replace Forward Euler with Trapezoidal** for all diff eqs, or reduce `dt` to 0.0001s (100× more substeps).

### Priority 2 — IEEE Compliance

6. **IEEET1:** Add SE(Efd) saturation, KF feedback, correct exciter transfer function.
7. **TGOV1:** Add valve rate limits, turbine damping Dt, initialize from PF.
8. **PSS:** Either implement true PSS2A (dual-input) or relabel as PSS1A. Fix lead-lag feed-through term. Add second lead-lag stage.
9. **Wind:** Add cut-out speed at 25 m/s to `DERManager.set_wind_output()`. Fix power curve formula.
10. **UFLS:** Implement multi-stage load shedding per IEGC.

### Priority 3 — Architecture

11. **Close the dynamics loop:** dynamics outputs → update pandapower gen → re-run PF.
12. **Merge duplicate DER models** — single source of truth in `der.py`.
13. **Increase dynamics substeps** or use event-driven stepping.
14. **Add IEEE 39-bus benchmark test** — compare PF results to published values.

---

## 10. Appendix — Full Bug Index

| ID | Category | Severity | One-Line Summary |
|----|----------|----------|-----------------|
| IEEE-01 | AVR | HIGH | Missing SE(Efd) saturation function |
| IEEE-02 | AVR | HIGH | Missing KF derivative feedback |
| IEEE-03 | AVR | HIGH | Exciter transfer function wrong |
| IEEE-04 | AVR | HIGH | Efd not initialized from PF |
| IEEE-05 | AVR | MEDIUM | Vref not initialized from PF |
| IEEE-06 | Governor | MEDIUM | Missing valve rate limiting |
| IEEE-07 | Governor | LOW | Missing turbine damping Dt |
| IEEE-08 | Governor | HIGH | Pg/Pm not initialized from PF |
| IEEE-09 | Governor | LOW | Droop formulation minor deviation |
| IEEE-10 | PSS | HIGH | Single-input, not dual-input PSS2A |
| IEEE-11 | PSS | MEDIUM | Missing ramp tracking filter |
| IEEE-12 | PSS | HIGH | Lead-lag DC gain = 4.67 instead of 1.0 |
| IEEE-13 | PSS | MEDIUM | Only one lead-lag stage (need two) |
| IEEE-14 | PSS | MEDIUM | Excessive effective gain |
| IEEE-15 | Topology | CRITICAL | Bus 0-index vs 1-index mismatch |
| IEEE-16 | Topology | LOW | Gen 1 H=500 effectively infinite bus |
| IEEE-17 | Topology | MEDIUM | Transformer taps not verified post-merge |
| IEEE-18 | Topology | MEDIUM | MVA base inconsistency dynamic ↔ PF |
| IEEE-19 | Topology | CRITICAL | ext_grid removal from Areas B/C |
| IEEE-20 | Protection | HIGH | No multi-stage UFLS |
| IEEE-21 | Protection | MEDIUM | No multi-stage OFGT |
| IEEE-22 | Protection | LOW | AGC deadband 0.02 vs IEGC 0.03 Hz |
| IEEE-23 | Protection | MEDIUM | Frequency trip delay too long |
| IEEE-24 | Protection | MEDIUM | No DER islanding detection |
| IEEE-25 | Wind | HIGH | No cut-out speed in DERManager |
| IEEE-26 | Wind | HIGH | Power curve shape incorrect |
| IEEE-27 | Wind | MEDIUM | No LVRT / reactive support |
| IEEE-28 | DER | MEDIUM | No voltage/freq ride-through |
| IEEE-29 | DER | MEDIUM | No volt-var / freq-watt |
| IEEE-30 | DER | MEDIUM | No inverter dynamics for PV |
| IEEE-31 | PF | MEDIUM | Losses calculation incorrect |
| IEEE-32 | PF | LOW | Hardcoded voltage thresholds |
| IEEE-33 | PF | LOW | NR Jacobian ignores transformer taps |
| N01 | Numerical | HIGH | Forward Euler marginally stable for AVR |
| N02 | Numerical | HIGH | AVR gain product → bang-bang |
| N03 | Numerical | MEDIUM | Washout signal path error |
| N04 | Numerical | MEDIUM | Float drift in load scale/restore |
| I01 | Init | CRITICAL | Eq_prime ignores angle |
| I02 | Init | HIGH | Exciter/governor hardcoded to 1.0 |
| I03 | Init | HIGH | Dynamic init doesn't set governor Pref |
| I04 | Init | MEDIUM | _battery_states never populated |
| I05 | Init | LOW | 0- vs 1-indexed bus in PF init mask |
| C01 | Control | CRITICAL | AGC adjustments never applied |
| C02 | Control | HIGH | Curtailment on input not output (cubic) |
| C03 | Control | HIGH | Load scale/restore doesn't cover DER loads |
| C04 | Control | MEDIUM | Per-area DER obs divides total by 3 |
| C05 | Control | MEDIUM | AGC participation factors not normalized |
| U01 | Units | CRITICAL | Solar panel_area undersized 250× |
| U02 | Units | MEDIUM | Diesel ramp rate unit mismatch |
| U03 | Units | MEDIUM | Battery SOC update ignores efficiency |
| T01 | Topology | CRITICAL | Tie-line bus 39 out of range |
| T02 | Topology | CRITICAL | ext_grid removal loses 2000 MVA |
| T03 | Topology | HIGH | Slack bus index may be off-by-one |
| T04 | Topology | MEDIUM | Area mappings not updated after DER add |
| P01 | PF | MEDIUM | Losses = gen − load (inaccurate) |
| P02 | PF | MEDIUM | Hardcoded violation thresholds |
| P03 | PF | LOW | Iteration count from _ppc may be zero |
| P04 | PF | LOW | NR Jacobian vs Ybus inconsistency |
| TS01 | Thread | HIGH | DR curtailment missing lock |
| TS02 | Thread | MEDIUM | Battery SOC update missing lock |
| TS03 | Thread | LOW | GridOrchestrator has no thread safety |
| A01 | Arch | MEDIUM | Duplicate DER models |
| A02 | Arch | MEDIUM | Two GridOrchestrator classes |
| A03 | Arch | MEDIUM | Deprecated GridManager broken method |
| A04 | Arch | LOW | Dead code standalone dynamics factory |
| A05 | Arch | LOW | Unclear energy_manager relationship |
| PERF01 | Perf | HIGH | 0.03% dynamics time coverage |
| PERF02 | Perf | MEDIUM | O(n²) Python Jacobian |
| PERF03 | Perf | LOW | Linear scan for DER lookup |

**Total: 5 CRITICAL · 14 HIGH · 16 MEDIUM · 8 LOW = 43 issues (plus 33 IEEE compliance violations, many overlapping)**

---

*End of Audit Report*
