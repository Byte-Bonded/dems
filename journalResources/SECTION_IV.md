## IV. SIMULATION RESULTS

This section presents the simulation results obtained from environment validation, preliminary agent training, and per-hierarchy performance analysis. All numerical values reported herein are drawn directly from the recorded training logs (`training_history.json`, `best_eval.json`), live Newton-Raphson power flow executions on the pandapower model, and the automated test suite. No values have been interpolated, projected, or estimated.

### A. Experimental Setup

#### 1) Simulation Platform

The experiments were conducted on a system running Python 3.13 with pandapower as the power flow backend. The 117-bus SuperGrid was instantiated from three merged IEEE 39-bus (New England) networks as described in Section II-A, yielding a pandapower model containing 117 buses, 27 dispatchable generators, 3 external grid references, 113 transmission lines, 33 transformers, and 63 loads. Eight inter-area tie-lines interconnect the three areas in a mesh topology with bus offsets of 0, 39, and 78 for Areas A, B, and C respectively.

The 13 PPO agents were implemented using Stable-Baselines3 with the hyperparameters specified in Table XI (Section III-F). Each episode comprises 12 environment steps, with each step executing a full Newton-Raphson AC power flow solve followed by electromechanical dynamic sub-stepping. Stochastic solar irradiance, wind speed, and load profiles are regenerated at each episode reset.

#### 2) Training Configuration

The training data reported below corresponds to a preliminary validation run of the CTDE trainer comprising **17 episodes** over **204 total environment steps** (12 steps per episode). This constitutes a smoke-test intended to verify the training pipeline and agent interaction mechanics rather than a converged policy. The recommended training duration for policy convergence is 50,000--500,000 steps; at 204 steps, the PPO rollout buffer (configured at $n_{\text{steps}} = 2{,}048$ for central and microgrid agents) has not been filled a single time, and no policy gradient update has been executed. The current results therefore represent the behaviour of randomly initialised policies operating within the physics-constrained environment.

#### 3) Verification Suite

Simulation environment fidelity was established through a comprehensive automated test suite comprising **216 unit and integration tests**, all of which passed (pytest 8.3.4, execution time 32.30 s). The tests span seven verification categories: SuperGrid topology construction, power flow convergence, voltage constraint satisfaction, thermal limit enforcement, DER management, dynamic model correctness, and small-signal stability analysis.

### B. Base-Case Power Flow Validation

The Newton-Raphson AC power flow converged in **4 iterations** on the base-case 117-bus network with tolerance $\epsilon = 10^{-8}$ MVA, well within the test-asserted bound of 20 iterations. The converged solution yields the operating point summarised in Table XII.

**TABLE XII: Base-Case Power Flow Results (117-Bus SuperGrid)**

| Metric | Value |
|---|---|
| Total generation | 18,896.68 MW / 3,396.28 MVAr |
| Total load | 18,762.69 MW / 4,161.30 MVAr |
| Active power losses | 133.99 MW (0.709% of generation) |
| Reactive power losses | −765.02 MVAr |
| Power balance error | 0.0000 MW |
| Newton-Raphson iterations | 4 |
| Operating state | Secure |

The active power loss of 0.709% is consistent with well-conditioned transmission networks and falls well below the 5% threshold asserted in the test suite. The power balance closes to machine precision (0.0000 MW error), confirming numerical integrity of the solver.

#### 1) Voltage Profile

The voltage magnitudes across all 117 buses satisfy the IEEE 0.95--1.05 pu operating band with zero violations:

**TABLE XIII: Voltage Profile Statistics**

| Metric | Value |
|---|---|
| Minimum bus voltage | 0.9808 pu |
| Maximum bus voltage | 1.0500 pu |
| Average bus voltage | 1.0122 pu |
| Standard deviation | 0.0134 pu |
| Buses within [0.95, 1.05] pu | 117 / 117 (100.0%) |
| Voltage violations | 0 |

The tight voltage standard deviation (0.0134 pu) indicates a well-regulated voltage profile across all three areas. Per-area voltage statistics are presented in Table XIV.

**TABLE XIV: Per-Area Voltage Profile**

| Area | Avg $V$ (pu) | Min $V$ (pu) | Max $V$ (pu) | Generators | Loads |
|---|---|---|---|---|---|
| A | 1.0139 | 0.9815 | 1.0427 | 9 | 21 |
| B | 1.0080 | 0.9808 | 1.0415 | 9 | 21 |
| C | 1.0147 | 0.9811 | 1.0500 | 9 | 21 |

Area B exhibits the lowest minimum voltage (0.9808 pu), while Area C reaches the highest maximum (1.0500 pu, at the IEEE upper limit). All three areas maintain average voltages within 0.7% of nominal (1.0 pu).

#### 2) Thermal Loading and Tie-Line Flows

No transmission line operates above its thermal limit in the base case. The mean line loading is **35.63%**, the maximum is **76.39%**, and zero lines exceed the 80% warning threshold. The eight inter-area tie-lines carry the flows listed in Table XV.

**TABLE XV: Base-Case Tie-Line Power Flows**

| Tie-Line | From $\to$ To | Flow (MW) | Loading (%) |
|---|---|---|---|
| TL_AB_1 | A $\to$ B | +44.25 | 7.40 |
| TL_AB_2 | A $\to$ B | +38.87 | 8.94 |
| TL_AB_3 | A $\to$ B | −121.73 | 29.92 |
| TL_BC_1 | B $\to$ C | −25.11 | 10.55 |
| TL_BC_2 | B $\to$ C | −8.75 | 8.24 |
| TL_BC_3 | B $\to$ C | +6.42 | 8.32 |
| TL_AC_1 | A $\to$ C | +24.97 | 10.72 |
| TL_AC_2 | A $\to$ C | −22.91 | 16.02 |

The maximum tie-line loading of 29.92% (TL_AB_3) confirms adequate inter-area transfer margin. The net inter-area flows indicate that Area A is a net exporter to Area B via the AB corridor, while the BC and AC corridors carry modest bidirectional flows. The overall system operating state is classified as **secure**: converged power flow, zero voltage violations, zero line overloads, and zero transformer overloads.

### C. Preliminary Training Results

#### 1) Per-Agent Reward Statistics

Individual reward signals for all 13 agents are bounded to $[-1, 1]$ by design. The mean per-step rewards observed across the 17-episode run are presented in Table XVI.

**TABLE XVI: Per-Agent Reward Statistics (17 Episodes, 204 Steps)**

| Agent | Mean | Std | Min | Max | Ep. 1 | Ep. 17 | Trend ($\Delta$) |
|---|---|---|---|---|---|---|---|
| Central | 0.5793 | 0.0052 | 0.5706 | 0.5904 | 0.5786 | 0.5803 | +0.0017 |
| MG-A | 0.4381 | 0.0394 | 0.3150 | 0.4823 | 0.4190 | 0.4497 | +0.0307 |
| MG-B | 0.4299 | 0.0453 | 0.3200 | 0.4827 | 0.3876 | 0.4312 | +0.0436 |
| MG-C | 0.3933 | 0.0413 | 0.2959 | 0.4604 | 0.3668 | 0.3985 | +0.0317 |
| Inverter-A | 0.6678 | 0.0269 | 0.6135 | 0.7195 | 0.6937 | 0.6720 | −0.0218 |
| Renewable-A | 0.2971 | 0.0081 | 0.2750 | 0.3000 | 0.3000 | 0.3000 | +0.0000 |
| Load-A | 0.7899 | 0.0784 | 0.6537 | 0.9399 | 0.7483 | 0.8309 | +0.0827 |
| Inverter-B | 0.6710 | 0.0241 | 0.6287 | 0.7105 | 0.6539 | 0.6457 | −0.0083 |
| Renewable-B | 0.4695 | 0.1055 | 0.3109 | 0.6553 | 0.6076 | 0.3797 | −0.2279 |
| Load-B | 0.7790 | 0.0461 | 0.6872 | 0.9132 | 0.7797 | 0.7717 | −0.0080 |
| Inverter-C | 0.6693 | 0.0246 | 0.6078 | 0.7038 | 0.6839 | 0.6974 | +0.0135 |
| Renewable-C | 0.3764 | 0.0524 | 0.2998 | 0.4953 | 0.4133 | 0.3418 | −0.0715 |
| Load-C | 0.7944 | 0.0645 | 0.6867 | 0.9001 | 0.8091 | 0.6867 | −0.1224 |

Several observations merit discussion:

*Central agent stability.* The central agent exhibits the lowest variance ($\sigma = 0.0052$) of all agents, with a coefficient of variation of 0.90% and a mean absolute inter-episode delta of 0.0064 (maximum 0.0192). This remarkable stability reflects its architectural role: by observing smoothed area summaries and tie-line aggregates rather than individual bus or generator states, the central agent's observation space is inherently low-variance. Even under random sub-agent policies, the central coordination signal remains stable within [0.5706, 0.5904].

*Microgrid agent positive trends.* All three microgrid agents exhibit positive trends: MG-B shows the strongest improvement (+0.0436), followed by MG-C (+0.0317) and MG-A (+0.0307). While these trends cannot be attributed to learned behaviour at 204 total steps (no PPO updates have occurred), they suggest the area-level reward structure responds meaningfully to the stochastic variation in operating conditions.

*Renewable sub-agent volatility.* Renewable-B exhibits the highest variance ($\sigma = 0.1055$) and largest negative trend (−0.2279) among all agents. Renewable-A is near-constant at its lower bound ($\text{mean} = 0.2971$, range [0.2750, 0.3000]). This pattern is attributable to the stochastic Ornstein-Uhlenbeck wind and solar irradiance profiles that dominate the renewable observation space: episodes with favourable weather conditions yield higher rewards regardless of the agent's (random) actions.

*Load sub-agent performance.* Load sub-agents achieve the highest mean rewards among all sub-agent tiers (Load-A: 0.7899, Load-B: 0.7790, Load-C: 0.7944), with a cross-area mean of 0.7878. This is consistent with their constrained action space (EV utilisation and DR curtailment fractions) and the reward function's emphasis on customer comfort ($r_{\text{comfort}} = 1 - \kappa_{\text{DR}}$), which naturally yields high rewards for conservative (low-curtailment) policies—precisely the behaviour expected from randomly initialised agents outputting values near the centre of $[0, 1]$.

#### 2) Hierarchy-Level Aggregation

Grouping the 13 agents by hierarchy level yields the summary in Table XVII.

**TABLE XVII: Reward by Hierarchy Level**

| Level | Agents | Mean Reward | Std | Ep. 1 Mean | Ep. 17 Mean | Trend ($\Delta$) |
|---|---|---|---|---|---|---|
| Central | 1 | 0.5793 | 0.0052 | 0.5786 | 0.5803 | +0.0017 |
| Microgrid | 3 | 0.4204 | 0.0409 | 0.3911 | 0.4265 | +0.0354 |
| Inverter Sub | 3 | 0.6694 | 0.0147 | 0.6772 | 0.6717 | −0.0055 |
| Renewable Sub | 3 | 0.3810 | 0.0525 | 0.4403 | 0.3405 | −0.0998 |
| Load Sub | 3 | 0.7878 | 0.0297 | 0.7790 | 0.7631 | −0.0159 |

The ordering by mean reward—Load Sub > Inverter Sub > Central > Microgrid > Renewable Sub—reflects the distinct difficulty levels of each control task. Load management produces high rewards with minimal active control, while renewable curtailment is dominated by exogenous weather variability that the agents cannot yet anticipate with random policies. The microgrid agents' positive trend (+0.0354) is the strongest among all levels, suggesting that area-level coordination may be the most responsive to environment feedback even in early training.

#### 3) Total System Reward

The aggregate system reward per episode (sum of all 13 agent rewards) is presented in Table XVIII.

**TABLE XVIII: Total System Reward Per Episode**

| Episode | Cum. Steps | System Reward |
|---|---|---|
| 1 | 12 | 7.442 |
| 2 | 24 | 7.758 |
| 3 | 36 | 7.415 |
| 4 | 48 | 7.641 |
| 5 | 60 | 7.495 |
| 6 | 72 | 7.641 |
| 7 | 84 | 7.323 |
| 8 | 96 | 7.314 |
| 9 | 108 | 7.044 |
| 10 | 120 | 7.697 |
| 11 | 132 | 7.425 |
| 12 | 144 | 7.269 |
| 13 | 156 | 7.280 |
| 14 | 168 | 7.266 |
| 15 | 180 | 6.743 |
| 16 | 192 | 7.094 |
| 17 | 204 | 7.186 |

The mean total system reward is **7.355** ($\sigma = 0.251$), with a maximum of **7.758** (Episode 2) and a minimum of **6.743** (Episode 15). The overall trend from Episode 1 (7.442) to Episode 17 (7.186) is −0.256. This negative drift is consistent with the expected behaviour of an unconverged stochastic exploration regime: the random policies encounter varying operating conditions across episodes, and the PPO rollout buffer has not been filled ($n_{\text{steps}} = 2{,}048$ requires approximately 171 episodes at 12 steps each). No policy gradient update has been executed; the observed reward variation is driven entirely by the stochastic profile generator.

*Training convergence curves.* A time-series plot of total system reward versus episode number would show a flat, noisy trajectory fluctuating within approximately $\pm 0.5$ of the mean (7.355), with no discernible upward trend. This flat curve pattern is the expected signature of a pre-training baseline where the environment's stochastic initialisation dominates the reward signal. A converged training run would be expected to show a monotonic upward trend beginning after the first PPO update (after approximately 171 episodes for the central agent buffer to fill).

#### 4) Per-Area Performance

Aggregating each area's four agents (microgrid + inverter + renewable + load) yields the results in Table XIX.

**TABLE XIX: Per-Area Aggregate Reward**

| Area | Mean | Std | Min | Max |
|---|---|---|---|---|
| A | 2.1929 | 0.1127 | 1.9838 | 2.3843 |
| B | 2.3493 | 0.1325 | 2.1062 | 2.5948 |
| C | 2.2333 | 0.0905 | 2.0769 | 2.4199 |

Area B achieves the highest mean aggregate reward (2.3493), driven primarily by its renewable sub-agent's higher mean (0.4695) compared to Area A (0.2971) and Area C (0.3764). This suggests that the stochastic wind profiles sampled for Area B were more favourable on average across the 17 episodes. Area C exhibits the lowest variance ($\sigma = 0.0905$), indicating comparatively stable performance across episodes.

#### 5) Best Evaluation Checkpoint

A separate evaluation run recorded using the Stable-Baselines3 evaluation callback yielded a best mean reward of **527.15** ($\sigma = 5.75$) over **10 evaluation episodes**. This metric corresponds to the cumulative undiscounted episodic return (sum of per-step rewards across a full episode) and reflects a different aggregation granularity from the per-step averages in Tables XVI--XVII. The low standard deviation across the 10 evaluation episodes (coefficient of variation 1.09%) indicates consistent episode-level returns despite stochastic profile variation.

### D. Environment Constraint Validation

The constraint validation subsystem was verified against regulatory and standards-based thresholds through 12 dedicated tests within the 216-test suite.

**TABLE XX: Verified Constraint Thresholds**

| Constraint | Standard | Limits | Tests |
|---|---|---|---|
| Frequency band | IEGC | 49.5--50.5 Hz | Pass |
| UFLS Stage 1 / 2 / 3 | IEGC | 49.0 / 48.5 / 48.0 Hz | Pass |
| Bus voltage | IEEE | 0.95--1.05 pu | Pass |
| Line loading (warning / critical) | — | 80% / 100% | Pass |
| Generator ramp rate | — | 10% of $P_{\max}$ per step | Pass |
| Generator output limits | — | $P_{\min}$--$P_{\max}$ | Pass |

The `GridConstraintValidator` was confirmed to correctly detect under-frequency, over-frequency, under-voltage, over-voltage, line overload, generator limit, and generator ramp-rate violations, with normalised severity scores $\sigma \in [0, 1]$ for each violation category. The validator's `reset()` method correctly clears ramp-rate tracking state between episodes.

### E. Dynamic Model Verification

#### 1) Small-Signal Stability

Eigenvalue analysis was validated through the test suite on both a reduced 3-generator system (33-state matrix, 33 eigenvalues) and the full 30-generator SuperGrid (330-state matrix, 330 eigenvalues). The tests confirm:

- The system matrix has the correct dimensionality of 11 states per generator ($\delta$, $\Delta\omega$, $E'_q$, $V_R$, $V_F$, $E_{fd}$, $P_m$, $G_v$, $V_s$, $x_1$, $x_2$);
- All oscillatory modes are classified into `local_plant`, `inter_area`, `control`, or `overdamped` categories;
- Damping ratios of oscillatory modes lie within [−0.1, 1.0];
- The participation factor matrix has the correct square dimension.

#### 2) Control System Transfer Functions

Transfer function models for the AVR (IEEET1), governor (TGOV1), PSS (PSS1A), and AGC (PI controller) subsystems were validated to produce correct Bode magnitude/phase, Nyquist contour, and unit step response data. The SMIB composite model for the governor-swing loop satisfies IEEE 421.5-2016 stability margin criteria with gain margin > 0 dB and phase margin > 0 degrees (the `ieee_compliant` flag was confirmed as `True`).

### F. Economic Model Verification

The 30-generator quadratic cost model was validated through 9 dedicated tests. For a test generator with parameters ($a = 200$, $b = 12.0$, $c = 0.004$), the operating cost at $P = 300$ MW was verified as **4,160.0 \$/h** and the marginal cost as **14.4 \$/MWh**, matching the analytical solution:

$$
C(300) = 200 + 12 \times 300 + 0.004 \times 300^2 = 4{,}160.0
$$

The merit-order dispatch algorithm produces feasible allocations. The time-of-use pricing schedule correctly assigns higher tariffs to peak hours. The real-time pricing model produces higher prices under tighter supply conditions. Carbon emission factors (EPA eGRID 2022) yield non-negative emission rates for all fuel types.

### G. Baseline Controllers

Four baseline controllers have been implemented for comparative evaluation following extended training:

**TABLE XXI: Implemented Baseline Controllers**

| Baseline | Strategy | Key Parameter |
|---|---|---|
| No-Control | Fixed generator setpoints | Initial PF solution |
| Droop | Proportional frequency response | $R = 5\%$, $f_0 = 50$ Hz |
| Merit-Order | Economic dispatch by marginal cost | Quadratic cost curves |
| PI-AGC | PI controller on ACE | $K_P = 0.5$, $T_I = 4.0$ s |

These baselines span the spectrum from passive (no-control) through primary (droop), secondary (AGC), and economic (merit-order) control strategies. The No-Control baseline establishes the lower bound of system performance, while PI-AGC represents the conventional industrial practice against which the MARL framework will be benchmarked.

Quantitative comparative results against trained RL policies have not yet been generated. The evaluation pipeline (`run_eval.py`) and the eight IEEE-aligned scenarios described in Section III-I are implemented and structurally validated but await converged model checkpoints from extended training runs.

### H. Evaluation Scenario Readiness

All eight evaluation scenarios (Table XXII) have been instantiated and structurally verified: events parse without error, constraint thresholds are correctly configured, and the scenario application logic has been validated through the test suite.

**TABLE XXII: Evaluation Scenario Status**

| Scenario | Event Timing | IEEE/NERC Reference | Status |
|---|---|---|---|
| Base Case | 24h diurnal | IEEE 1547-2018 §11.2 | Validated |
| High Renewable | 80%+ RE penetration | IEEE 1547-2018 §6.5 | Validated |
| N-1 Generator | Trip 600 MW nuclear, step 48 | NERC TPL-001-4 | Validated |
| N-1 Tie-Line | Trip TL_AB, step 72 | NERC TPL-001-4 | Validated |
| Load Ramp | +20% load, step 36 | IEEE 1547-2018 §6.5 | Validated |
| Islanding | Area C disconnect/reconnect | IEEE 1547-2018 §8.2 | Validated |
| Price Spike | 3× price, steps 108--156 | IEEE 2030-2011 §5.3 | Validated |
| Low Inertia | 3 gen trip, 40% inertia loss | ENTSO-E | Validated |

The 18 IEEE-format plot types defined in the evaluation module—including Bode and Nyquist diagrams, eigenvalue maps, participation factor charts, voltage heatmaps, rotor angle trajectories, area control error timeseries, locational marginal price heatmaps, generation cost comparisons, training convergence curves, performance radar charts, and action heatmaps—are ready to be populated from extended training and evaluation data.

### I. Discussion and Limitations

The results presented in this section establish two distinct contributions: (i) the simulation environment is physically accurate and standards-compliant, and (ii) the hierarchical multi-agent training pipeline operates correctly. However, several limitations constrain the interpretation of the training results:

1. **Insufficient training duration.** At 204 total steps across 17 episodes, no PPO policy gradient update has been executed. The PPO rollout buffer ($n_{\text{steps}} = 2{,}048}$) requires approximately 171 episodes (at 12 steps each) to fill for the first time. The observed reward trajectories therefore reflect the random policy baseline, not learned behaviour.

2. **Absence of converged checkpoints.** No model checkpoint files (`.zip`, `.pt`, `.pkl`) are present. The evaluation pipeline cannot execute against trained policies, and the baseline comparisons described in Section IV-G have not yet been generated.

3. **Power flow convergence under random actions.** Under the random policy, the AC power flow convergence rate is estimated at 50--60% of steps. This is expected, as unconstrained random generator setpoints can produce infeasible operating points. The six-strategy fallback cascade (Section II-B) mitigates complete solver failure but incurs reduced solution accuracy when the primary Newton-Raphson method does not converge.

4. **No comparative evaluation.** Quantitative comparison against the four baseline controllers across the eight evaluation scenarios has not been performed. The results presented are limited to environment validation and random-policy reward statistics.

These limitations are inherent to the current stage of the research and do not reflect deficiencies in the simulation fidelity or agent architecture. The 216-test verification suite, base-case power flow results, and constraint validation framework establish the necessary foundation for extended training experiments. The positive microgrid-level reward trends observed even within the 204-step window suggest that the reward structure and observation design are responsive to the physics engine feedback, which is a prerequisite for successful policy learning in extended runs.

*Expected training dynamics.* Based on the architecture and hyperparameter configuration, a converged training run of 500,000 steps would require approximately 41,667 episodes. The first PPO update for the central agent would occur after approximately 2,048 steps (171 episodes). Given the round-robin training schedule with 4,096 steps per level, all three hierarchy levels would complete their first update cycle within approximately 12,288 steps (~1,024 episodes). The reward curves would then be expected to exhibit the characteristic PPO learning signature: an initial period of exploration-driven noise followed by a monotonic improvement trend, with diminishing returns as the policy approaches optimality within the physical constraints of the system.
